cmake_minimum_required(VERSION 3.0.0)
project(unconstrained_optimisation)

include_directories(include)

add_library(unconstrained_optimisation)

target_sources(unconstrained_optimisation
PRIVATE
    src/gauss_newton.cc
)

target_link_libraries(unconstrained_optimisation
PUBLIC
    Boost::boost
    optimisation_problme
)

target_include_directories(unconstrained_optimisation
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(
    TARGETS
    unconstrained_optimisation
)

install(DIRECTORY include/ DESTINATION "$(CMAKE_INSTALL_INCLUDEDIR)")