cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(LegionResilience)

find_package(Legion REQUIRED)

enable_testing()

add_subdirectory(src)
