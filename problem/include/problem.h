#pragma once

#include <iostream>
#include <memory>

#include "eigen3/Eigen/Core"
namespace optimisation {

class VectorFunction {
public:
  virtual Eigen::Matrix<double, Eigen::Dynamic, 1>
  operator()(const double *vars) const = 0;
};

class Jacobian {
public:
  virtual Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  operator()(const double *vars) const = 0;
};

// Dimension of the vector valued function f is m
// Dimension of optimisation variable X is n
class LeastSquareProblem {
public:
  LeastSquareProblem(const std::shared_ptr<VectorFunction> &func,
                     const std::shared_ptr<Jacobian> &jacobian,
                     const int size_of_x, double *x)
      : func_(func), jacobian_(jacobian), size_of_x_(size_of_x), x_(x) {}

  // f: R(n) ---> R(m)
  // F: (1/2) * ||f||^2 (CostFunc)
  Eigen::Matrix<double, Eigen::Dynamic, 1> EvaluateVectorFunction() const;
  double EvaluateCostFunction() const;

  // Jacobian return a m-by-n matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  EvaluateJacobian() const;

  void UpdateVariable(const Eigen::Matrix<double, Eigen::Dynamic, 1> &h);

private:
  std::shared_ptr<VectorFunction> func_;
  std::shared_ptr<Jacobian> jacobian_;
  int size_of_x_;
  double *x_;
};

} // namespace optimisation