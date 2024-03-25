#include "example_problem.h"
#include "gauss_newton.h"
#include "problem.h"

#include <iostream>

int main() {
  double lambda = 2;
  double numer_of_iteration = 100;

  double solution = 10;
  optimisation::LeastSquareProblem problem(
      std::make_shared<optimisation::ExampleVectorFunction>(lambda),
      std::make_shared<optimisation::ExampleJacobian>(lambda), 1, &solution);

  unconstrained::GaussianNewtonSolve(problem, numer_of_iteration);
  std::cout << "solution: " << solution << std::endl;

  return 0;
}