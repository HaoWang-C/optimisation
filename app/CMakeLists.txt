add_executable(solve_test
solve_test.cc)

target_link_libraries(solve_test
PRIVATE
    Boost::boost
    optimisation_problme
    unconstrained_optimisation
    )

install(
    TARGETS
    solve_test
    EXPORT unconstrained_optimisation
)

add_executable(sprint_problem_test
sprint_problem_test.cc)

target_link_libraries(sprint_problem_test
PRIVATE
    Boost::boost
    gflags
    optimisation_problme
    unconstrained_optimisation
    )

install(
    TARGETS
    sprint_problem_test
    EXPORT unconstrained_optimisation
)