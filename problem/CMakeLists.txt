cmake_minimum_required(VERSION 3.0.0)
project(optimisation_problme)

include_directories(include)

add_library(optimisation_problme)

target_sources(optimisation_problme
PRIVATE
    src/problem.cc
)

target_link_libraries(optimisation_problme
PUBLIC
    Boost::boost
)

target_include_directories(optimisation_problme
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(
    TARGETS
    optimisation_problme
)

install(DIRECTORY include/ DESTINATION "$(CMAKE_INSTALL_INCLUDEDIR)")