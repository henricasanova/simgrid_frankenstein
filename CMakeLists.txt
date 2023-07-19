cmake_minimum_required(VERSION 3.2)
message(STATUS "Cmake version ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}")

project("simgrid_frankenstein")

add_definitions("-Wall -Wno-unused-variable -Wno-unused-private-field")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

set(CMAKE_BUILD_TYPE Debug)

find_package(SimGrid REQUIRED)

# include directories for dependencies and WRENCH libraries
include_directories(src/ ${SimGrid_INCLUDE_DIR}/include /usr/local/include /opt/local/include)

# source files
set(SOURCE_FILES
        ./main.cpp
       )

# generating the executable
add_executable(simgrid_frankenstein ${SOURCE_FILES})
set_property(TARGET simgrid_frankenstein PROPERTY CXX_STANDARD 17)

target_link_libraries(simgrid_frankenstein
                       ${SimGrid_LIBRARY}
                      )
install(TARGETS simgrid_frankenstein DESTINATION bin)

