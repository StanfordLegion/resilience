cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(LegionResilience)

find_package(Legion REQUIRED)

include(GNUInstallDirs)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
