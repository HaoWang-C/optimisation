#pragma once

#include "problem.h"
namespace optimisation {

class ExampleVectorFunction : public VectorFunction {
public:
  ExampleVectorFunction(const double lambda) : lambda_(lambda) {}

  Eigen::Matrix<double, Eigen::Dynamic, 1>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 2, 1> f_vector;
    f_vector << vars[0] + 1, lambda_ * vars[0] * vars[0] + vars[0] - 1;
    return f_vector;
  }

private:
  double lambda_;
};

class ExampleJacobian : public Jacobian {
public:
  ExampleJacobian(const double lambda) : lambda_(lambda) {}

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 2, 1> jacobian;
    jacobian << 1, (2 * lambda_ * vars[0] + 1);
    return jacobian;
  }

private:
  double lambda_;
};

} // namespace optimisation