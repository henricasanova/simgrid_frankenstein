#include <random>
#include <simgrid/s4u.hpp>
#include <smpi/smpi.h>
#include <vector>

namespace sg4 = simgrid::s4u;

constexpr double MB    = 1000000;
constexpr double GB    = 1000000000;
constexpr double GFLOP = 1000000000;

XBT_LOG_NEW_DEFAULT_CATEGORY(mwe, "Minimum Working Example");

struct WorkUnit {
  int num_iterations;
  double num_bytes;
  double num_flops;

  WorkUnit(int num_iterations, double num_bytes, double num_flops)
      : num_iterations(num_iterations), num_bytes(num_bytes), num_flops(num_flops) {}
};

class Coordinator {
public:
  void operator()()
  {

    auto my_mailbox = sg4::Mailbox::by_name("coordinator_mb");

    auto num_workers = sg4::Engine::get_instance()->get_netzone_root()->get_children().size() - 2;

    std::deque<WorkUnit*> todo;

    // Create workunits
    for (int i = 0; i < 10; i++) {
      todo.push_front(new WorkUnit(1, 10 * MB, 2 * GFLOP));
    }
    // Add "poison pills" at the end, one per worker
    for (unsigned int i = 0; i < num_workers; i++) {
      todo.push_front(new WorkUnit(0, 0, 0));
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
  void operator()()
  {
    auto cluster_hosts = sg4::this_actor::get_host()->get_englobing_zone()->get_all_hosts();
    auto db_host       = sg4::Host::by_name("database.org");
    auto db_disk       = db_host->get_disks().front();
    auto my_mailbox    = sg4::Mailbox::by_name("mb_" + sg4::this_actor::get_name());

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
      std::string smpi_program_name = "MPI_Job_" + sg4::this_actor::get_name();
      SMPI_app_instance_start(
          smpi_program_name.c_str(),
          [work_unit_spec, db_host, db_disk]() {
            MPI_Init();
            int num_procs, my_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

            if (my_rank == 0)
              XBT_INFO("MPI Job starting with %d processes", num_procs);

            void* data = SMPI_SHARED_MALLOC(work_unit_spec->num_bytes * num_procs);
            for (int iteration = 0; iteration < work_unit_spec->num_iterations; iteration++) {
              sg4::this_actor::execute(work_unit_spec->num_flops);

              MPI_Alltoall(data, work_unit_spec->num_bytes, MPI_CHAR, data, work_unit_spec->num_bytes, MPI_CHAR,
                           MPI_COMM_WORLD);

              if (my_rank == 0) {
                XBT_INFO("Rank 0 doing database upload");
                auto io_activity = db_disk->io_init(1 * MB, sg4::Io::OpType::READ);
                auto communication_activity =
                    sg4::Comm::sendto_async(sg4::this_actor::get_host(), db_host, 1 * MB)->add_successor(io_activity);
                io_activity->start();
                communication_activity->wait();
                XBT_INFO("Rank 0 done with database upload");
              }
            }
            SMPI_SHARED_FREE(data);
            MPI_Finalize();
            if (my_rank == 0) {
              XBT_INFO("MPI Job finishing");
            }
          },
          cluster_hosts);

      SMPI_app_instance_join(smpi_program_name);
      XBT_INFO("MPI Job on %ld compute nodes completed.", cluster_hosts.size());
      delete work_unit_spec;
    }
    XBT_INFO("Worker terminating.");
  }
};

static sg4::NetZone* create_cluster(const sg4::NetZone* root, const std::string& suffix, const int num_hosts)
{
  auto* cluster = sg4::create_star_zone("cluster" + suffix)->set_parent(root);

  /* create gateway */
  cluster->set_gateway(cluster->create_router("cluster" + suffix + "-router"));

  /* create the backbone link */
  auto* backbone = cluster->create_link("backbone" + suffix, "100Gbps")->set_latency("100us");

  /* create all hosts and connect them to outside world */
  for (int i = 0; i < num_hosts; i++) {
    std::string name = "host-" + std::to_string(i) + suffix;
    /* create host */
    const auto* host = cluster->create_host(name, "1Gf");
    /* create link */
    const auto* link = cluster->create_link(name + "_link", "10Gbps")->set_latency("10us");
    /* add route between host and any other host */
    cluster->add_route(host, nullptr, {link, backbone});
  }

  cluster->seal();
  return cluster;
}

int main(int argc, char** argv)
{
  auto engine = new sg4::Engine(&argc, argv);

  // Create the platform
  auto* root = sg4::create_full_zone("world");

  // Create a coordinator zone/host
  auto coordinator_zone = sg4::create_full_zone("Coordinator")->set_parent(root);
  coordinator_zone->create_host("coordinator.org", "1Gf");
  coordinator_zone->seal();

  // Create a database zone/host
  auto database_zone = sg4::create_full_zone("Database")->set_parent(root);
  database_zone->create_host("database.org", "1Gf")->create_disk("db", "100MBps", "50MBps");
  database_zone->seal();

  // Create a single link as a simple abstraction of the whole wide-area network
  auto* internet = root->create_link("internet", "200MBps")->set_latency("1ms");

  // Create three clusters
  std::vector<int> cluster_sizes = {16, 32, 40};
  int i = 0;
  for (auto size : cluster_sizes) {
    auto* cluster = create_cluster(root, ".cluster" + std::to_string(i++) + ".org", size);
    root->add_route(coordinator_zone, cluster, {internet});
    root->add_route(database_zone, cluster, {internet});
  }

  root->seal();

  // Create the coordinator and worker actors
  sg4::Actor::create("Coordinator", sg4::Host::by_name("coordinator.org"), Coordinator());
  for (unsigned int i = 0; i < cluster_sizes.size(); i++) {
    sg4::Actor::create("Worker" + std::to_string(i), sg4::Host::by_name("host-0.cluster" + std::to_string(i) + ".org"),
                       Worker());
  }

  SMPI_init();
  engine->run();
  SMPI_finalize();

  return 0;
}
