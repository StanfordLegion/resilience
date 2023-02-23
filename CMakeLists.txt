cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(LegionResilience)

find_package(Legion REQUIRED)
find_package(HDF5 REQUIRED)

include(GNUInstallDirs)

set(RESILIENCE_TEST_LAUNCHER "" CACHE STRING "launcher for test commands")

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
