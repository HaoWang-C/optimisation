#pragma once

#include <iostream>

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

// TODO: Alternative way is using functor
// Define a functor class
// class MyFunctor {
// public:
//     // Overload the function call operator ()
//     void operator()(int value) const {
//         std::cout << "Functor called with value: " << value << std::endl;
//     }
// };

// // Define a class that takes a functor as an input object
// class FunctionWrapper {
// public:
//     // Constructor taking a functor as argument
//     FunctionWrapper(const MyFunctor& func) : m_func(func) {}

//     // Method to execute the stored functor with an integer argument
//     void execute(int value) {
//         m_func(value);
//     }

// private:
//     // Functor member
//     MyFunctor m_func;
// };

namespace optimisation {

// Dimension of the function F is m
// Dimension of optimisation variable X is n
template <int m, int n> class LeastSquareProblem {
public:
  // f: R(n) ---> R(m)
  // F: (1/2) * ||f||^2 (CostFunc)
  virtual Eigen::Matrix<double, m, 1> func() const = 0;
  virtual double CostFunc() const = 0;

  // Jacobian return a m-by-n matrix
  virtual Eigen::Matrix<double, m, n> Jacobian_of_f() const = 0;

  virtual bool UpdateVariable(const std::vector<double> &delta_x) = 0;

  virtual ~LeastSquareProblem() {}
};

class ExampleProblem : public LeastSquareProblem<2, 1> {
public:
  ExampleProblem(const double x) : x_(x) {}

  Eigen::Matrix<double, 2, 1> func() const override;

  double CostFunc() const override;

  Eigen::Matrix<double, 2, 1> Jacobian_of_f() const override;

  bool UpdateVariable(const std::vector<double> &delta_x) override;

private:
  double x_;
};

} // namespace optimisation