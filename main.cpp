#include <simgrid/s4u.hpp>
#include <smpi/smpi.h>
#include <vector>
#include <random>

namespace sg4 = simgrid::s4u;

#define MB 1000000
#define GB 1000000000
#define GFLOP 1000000000

XBT_LOG_NEW_DEFAULT_CATEGORY(mwe, "Minimum Working Example");


struct WorkUnit {
    int num_iterations;
    int num_bytes;
    double num_flops;

    WorkUnit(int num_iterations, int num_bytes, double num_flops) :
            num_iterations(num_iterations), num_bytes(num_bytes), num_flops(num_flops) {}

};

class Coordinator {
public:
    void operator()() {

        auto my_mailbox = sg4::Mailbox::by_name("coordinator_mb");

        auto num_workers = sg4::this_actor::get_host()->get_englobing_zone()->get_parent()->get_children().size() - 2;

        std::deque<WorkUnit*> todo;

        // Create workunits
        for (int i=0; i < 10; i++) {
            todo.push_front(new WorkUnit(1, 10*MB, 2*GFLOP));
        }
        // Add "poison pills" at the end, one per worker
        for (int i=0; i < num_workers; i++) {
            todo.push_front(new WorkUnit(0,0,0));
        }

        // Main loop
        XBT_INFO("Coordinator starting: %ld workunits to do.", todo.size());
        while (not todo.empty()) {
            auto worker_mailbox = my_mailbox->get<sg4::Mailbox>();
            worker_mailbox->put(todo.back(), 128);
            todo.pop_back();
        }
        XBT_INFO("Coordinator terminating.");
    }
};

class Worker {

public:
    void operator()() {
        auto cluster_hosts = sg4::this_actor::get_host()->get_englobing_zone()->get_all_hosts();
        auto db_host = sg4::Host::by_name("database.org");
        auto db_disk = db_host->get_disks().at(0);
        auto my_mailbox = sg4::Mailbox::by_name("mb_" + sg4::this_actor::get_name());

        XBT_INFO("Worker starting on a cluster with %ld compute nodes.", cluster_hosts.size());
        // Main loop
        while (true) {
            // Ask the coordinator for work
            sg4::Mailbox::by_name("coordinator_mb")->put(my_mailbox, 32);
            // Wait for a work unit as a reply
            auto work_unit_spec = my_mailbox->get<WorkUnit>();
            if (work_unit_spec->num_iterations == 0) {
                break; // Poison pill
            }
            XBT_INFO("Received a workunit.");

            XBT_INFO("Starting MPI Job on %ld compute nodes.", cluster_hosts.size());
            auto barrier = simgrid::s4u::Barrier::create(1 + cluster_hosts.size());
            SMPI_app_instance_start(("MPI_Job_" + sg4::this_actor::get_name()).c_str(),
                                    [barrier, work_unit_spec, db_host, db_disk]() {
                                        MPI_Init();
                                        int num_procs, my_rank;
                                        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
                                        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

                                        if (my_rank == 0) {
                                            XBT_INFO("MPI Job starting with %d processes", num_procs);
                                        }
                                        void *data = SMPI_SHARED_MALLOC(work_unit_spec->num_bytes * num_procs);
                                        for (int iteration = 0; iteration < work_unit_spec->num_iterations; iteration++) {
                                            sg4::this_actor::execute(work_unit_spec->num_flops);

                                            MPI_Alltoall(data, work_unit_spec->num_bytes, MPI_CHAR, data, work_unit_spec->num_bytes, MPI_CHAR, MPI_COMM_WORLD);

                                            if (my_rank == 0) {
                                                XBT_INFO("Rank 0 doing database upload");
                                                auto io_activity   = db_disk->io_init(1*MB, sg4::Io::OpType::READ);
                                                auto communication_activity = sg4::Comm::sendto_init(sg4::this_actor::get_host(), db_host)->set_payload_size(1*MB)->add_successor(io_activity);
                                                io_activity->start();
                                                communication_activity->start()->wait();
                                                XBT_INFO("Rank 0 done with database upload");
                                            }
                                        }
                                        SMPI_SHARED_FREE(data);
                                        MPI_Finalize();
                                        if (my_rank == 0) {
                                            XBT_INFO("MPI Job finishing");
                                        }
                                        barrier->wait();
                                    },
                                    cluster_hosts);
            barrier->wait();
            XBT_INFO("MPI Job on %ld compute nodes completed.", cluster_hosts.size());
            delete work_unit_spec;
        }
        XBT_INFO("Worker terminating.");
    }
};


static std::tuple<sg4::NetZone*, simgrid::kernel::routing::NetPoint*> create_cluster(const sg4::NetZone* root,
                                                                                     const std::string& cluster_suffix,
                                                                                     const int num_hosts)
{
    auto* cluster = sg4::create_star_zone("cluster" + cluster_suffix);
    cluster->set_parent(root);

    /* create the backbone link */
    const sg4::Link* l_bb = cluster->create_link("backbone" + cluster_suffix, 100*GB)->set_latency(1e-4)->seal();

    /* create all hosts and connect them to outside world */
    for (int i=0; i < num_hosts; i++) {
        std::string hostname = "host-" + std::to_string(i)  + cluster_suffix;
        /* create host */
        const sg4::Host* host = cluster->create_host(hostname, 1*GFLOP);
        /* create UP link */
        const sg4::Link* l_up = cluster->create_link(hostname + "_up", 10*GB)->set_latency(1e-5)->seal();
        /* create DOWN link, if needed */
        const sg4::Link* l_down = l_up;
        if (i != 0) {
            l_down = cluster->create_link(hostname + "_down", 10*GB)->set_latency(1e-5)->seal();
        }
        sg4::LinkInRoute backbone{l_bb};
        sg4::LinkInRoute link_up{l_up};
        sg4::LinkInRoute link_down{l_down};

        /* add link UP and backbone for communications from the host */
        cluster->add_route(host->get_netpoint(), nullptr, nullptr, nullptr, {link_up, backbone}, false);
        /* add backbone and link DOWN for communications to the host */
        cluster->add_route(nullptr, host->get_netpoint(), nullptr, nullptr, {backbone, link_down}, false);
    }

    /* create router */
    auto router = cluster->create_router("router" + cluster_suffix);

    cluster->seal();
    return std::make_tuple(cluster, router);
}

int main(int argc, char **argv) {
    auto engine = new sg4::Engine(&argc, argv);

    // Create the platform
    auto* root = sg4::create_full_zone("AS0");

    // Create a coordinator zone/host
    auto coordinator_zone = sg4::create_full_zone("Coordinator")->set_parent(root);
    auto coordinator_host = coordinator_zone->create_host("coordinator.org", 1*GFLOP);

    // Create a database zone/host 
    auto database_zone = sg4::create_full_zone("Database")->set_parent(root);
    auto database_host = database_zone->create_host("database.org", 1*GFLOP);
    database_host->create_disk("db", 100*MB, 50*MB);

    // Create a single link as a simple abstraction of the whole wide-area network
    const sg4::Link* l = root->create_link("link1-2", 200*MB)->set_latency(1e-3)->seal();
    sg4::LinkInRoute internet{l};

    // Create two clusters
    std::vector<int> cluster_sizes = {16, 32, 40};
    for (int i=0; i < cluster_sizes.size(); i++) {
        auto cluster = create_cluster(root, ".cluster" + std::to_string(i) + ".org", cluster_sizes.at(i));
        root->add_route(coordinator_zone->get_netpoint(), std::get<0>(cluster)->get_netpoint(), coordinator_host->get_netpoint(), std::get<1>(cluster), {internet});
        root->add_route(database_zone->get_netpoint(), std::get<0>(cluster)->get_netpoint(), database_host->get_netpoint(), std::get<1>(cluster), {internet});
    }

    root->seal();

    // Create the coordinator and worker actors
    sg4::Actor::create("Coordinator", sg4::Host::by_name("coordinator.org"), Coordinator());
    for (int i=0; i < cluster_sizes.size(); i++) {
        sg4::Actor::create("Worker" + std::to_string(i), sg4::Host::by_name("host-0.cluster" + std::to_string(i) + ".org"), Worker());
    }

    SMPI_init();
    engine->run();
    SMPI_finalize();

    return 0;
}
