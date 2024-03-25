#pragma once

#include "eigen3/Eigen/Core"
// Design a optimisation problem then design a solver. This code design the
// problem. We need:
//    1. A cost function F that can be evaluated at x
//    2. The Jacobian that can be evaluated at x
//    3. The Hessian that can be evaluated at x

// Design:
//  1. The problem should be a "class".
//  2. The problem class should be able to calculate F J H.
//  3. The problem should also be able to access/update the optimisation
//  variable
//  4. The optimisation variable x should be a double value vector

// Note:
//  1. Only least square problem is considered in this initial work.

namespace optimisation {

// Dimension of the function F is m
// Dimension of optimisation variable X is n
template <int m, int n> class LeastSquareProblem {
public:
  // Cost function return a scalar
  virtual double Func() const = 0;

  // Jacobian return a m-by-n matrix
  virtual Eigen::Matrix<double, m, n> Jacobian() const = 0;

  // Hessian return a m-by-n matrix
  virtual Eigen::Matrix<double, m, n> Hessian() const = 0;

  virtual bool UpdateVariable() = 0;

  virtual ~LeastSquareProblem() {}
};

} // namespace optimisation