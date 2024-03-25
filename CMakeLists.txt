cmake_minimum_required(VERSION 3.0.0)
project(optimisation)

find_package(Boost REQUIRED)

add_subdirectory(unconstrained)
add_subdirectory(problem)
add_subdirectory(app)