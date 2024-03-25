add_executable(solve_test
solve_test.cc)

target_link_libraries(solve_test
PRIVATE
    Boost::boost
    optimisation_problme)

install(
    TARGETS
    solve_test
    EXPORT unconstrained_optimisation
)