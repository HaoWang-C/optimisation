#include "gauss_newton.h"
#include "problem.h"

#include <iostream>

int main() {
  optimisation::ExampleProblem ex_prob(3);

  std::cout << "f:\n" << ex_prob.func() << std::endl;
  std::cout << "F:\n" << ex_prob.CostFunc() << std::endl;
  std::cout << "Jacobian:\n" << ex_prob.Jacobian_of_f() << std::endl;

  unconstrained::GaussianNewtonSolve(ex_prob);

  return 0;
}