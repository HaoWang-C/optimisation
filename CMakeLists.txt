cmake_minimum_required(VERSION 3.0.0)
project(optimisation)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)


add_subdirectory(unconstrained)
add_subdirectory(problem)
add_subdirectory(app)