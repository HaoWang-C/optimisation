#include "example_problem.h"
#include "gauss_newton.h"
#include "problem.h"

#include <gflags/gflags.h>
#include <iostream>

DEFINE_int32(number_of_iteration, 100.0, "Number of iterations");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Large gap
  {
    // Initial solution
    double solution[3] = {1, 2, 3};
    std::vector<double> odometry_weight{1, 1, 1};
    double loop_closure_weight{1};
    double loop_closure_length{4};

    optimisation::LeastSquareProblem problem(
        std::make_shared<optimisation::SpringVectorFunction>(
            odometry_weight, loop_closure_length, loop_closure_weight),
        std::make_shared<optimisation::SpringJacobian>(odometry_weight,
                                                       loop_closure_weight),
        3, solution);

    unconstrained::GaussianNewtonSolve(problem, FLAGS_number_of_iteration);
    std::cout << "solution for large gap: ";
    optimisation::printArray(solution, 3);
    std::cout << std::endl;
  }

  // Small gap
  {
    // Initial solution
    double solution[6] = {1, 1.25, 1.5, 1.75, 2, 3};
    std::vector<double> odometry_weight{1, 4, 4, 4, 4, 1};
    double loop_closure_weight{1};
    double loop_closure_length{4};

    optimisation::LeastSquareProblem problem(
        std::make_shared<optimisation::SpringVectorFunctionSmallerConstraints>(
            odometry_weight, loop_closure_length, loop_closure_weight),
        std::make_shared<optimisation::SpringJacobianSmallerConstraints>(
            odometry_weight, loop_closure_weight),
        6, solution);

    unconstrained::GaussianNewtonSolve(problem, FLAGS_number_of_iteration);
    std::cout << "solution for small gap: ";
    optimisation::printArray(solution, 6);
    std::cout << std::endl;
  }
  return 0;
}