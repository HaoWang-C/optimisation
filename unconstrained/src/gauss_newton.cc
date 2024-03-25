#include "gauss_newton.h"

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace unconstrained {

// Gaussian-Newton use a linear model to approximate the f():
//      f(x+h) ~= l(h) := f(x) + J(x)*h
//      F(x+h) ~= L(h) := 1/2 * l(h)^t * l(h)
//                      = F(x) + h^t * J^t * f + 1/2 * h^t * J^t * J * h
// and the Gaussian-Newton step h_gn := argmin_h{ L(h) }.
// Solving h_gn = argmin{ L(h) } is easy - it is a quadratic programming. We can
// see the Hessian of L(h) is J^t * J which is symmetric and if it is also
// positive definite then L(h) has a global minimizer which can be found by
// solving the linear equation:
//      (J^t * J) * h_gn = -J^t * f
void GaussianNewtonSolve(optimisation::LeastSquareProblem &problem,
                         const int numer_of_iteration) {
  for (size_t i = 0; i < numer_of_iteration; i++) {
    // Step-1: Solve (J^t * J) * h_gn = -J^t * f
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian =
        problem.EvaluateJacobian();

    auto H_matrix_inverse = (jacobian.transpose() * jacobian).inverse();
    // std::cout << "H_matrix:\n" << H_matrix_inverse << std::endl;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b_vector =
        (-1) * jacobian.transpose() * problem.EvaluateVectorFunction();
    // std::cout << "b_vector:\n" << b_vector << std::endl;

    auto h_gn = H_matrix_inverse * b_vector;
    // std::cout << "h_gn:\n" << h_gn << std::endl;

    // Step-2: Update the optimisation variables
    problem.UpdateVariable(h_gn);
  }
}

} // namespace unconstrained
