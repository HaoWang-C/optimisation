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

class SpringVectorFunction : public VectorFunction {
public:
  SpringVectorFunction(const std::vector<double> &odometry_weights,
                       const double loop_closures_length,
                       const double loop_closure_weight)
      : odometry_weights_(odometry_weights),
        loop_closures_length_(loop_closures_length),
        loop_closure_weight_(loop_closure_weight) {}

  Eigen::Matrix<double, Eigen::Dynamic, 1>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 4, 1> f_vector;
    double f_1 = sqrt(odometry_weights_.at(0)) * (vars[0] - 1);
    double f_2 = sqrt(odometry_weights_.at(1)) * (vars[1] - vars[0] - 1);
    double f_3 = sqrt(odometry_weights_.at(2)) * (vars[2] - vars[1] - 1);
    double f_4 = sqrt(loop_closure_weight_) * (vars[2] - loop_closures_length_);

    f_vector << f_1, f_2, f_3, f_4;
    // std::cout << "f_vector: " << f_vector << std::endl;
    return f_vector;
  }

private:
  std::vector<double> odometry_weights_;

  // >= 4
  double loop_closures_length_;
  double loop_closure_weight_;
};

class SpringJacobian : public Jacobian {
public:
  SpringJacobian(const std::vector<double> &odometry_weights,
                 const double loop_closure_weight)
      : odometry_weights_(odometry_weights),
        loop_closure_weight_(loop_closure_weight) {}

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 4, 3> jacobian;
    jacobian.row(0) << sqrt(odometry_weights_.at(0)), 0.0, 0.0;
    jacobian.row(1) << (-1.0) * sqrt(odometry_weights_.at(1)),
        sqrt(odometry_weights_.at(1)), 0.0;
    jacobian.row(2) << 0.0, (-1.0) * sqrt(odometry_weights_.at(2)),
        sqrt(odometry_weights_.at(2));
    jacobian.row(3) << 0.0, 0.0, sqrt(loop_closure_weight_);

    // std::cout << "jacobian: " << jacobian << std::endl;

    return jacobian;
  }

private:
  std::vector<double> odometry_weights_;
  double loop_closure_weight_;
};

class SpringVectorFunctionSmallerConstraints : public VectorFunction {
public:
  SpringVectorFunctionSmallerConstraints(
      const std::vector<double> &odometry_weights,
      const double loop_closures_length, const double loop_closure_weight)
      : odometry_weights_(odometry_weights),
        loop_closures_length_(loop_closures_length),
        loop_closure_weight_(loop_closure_weight) {}

  Eigen::Matrix<double, Eigen::Dynamic, 1>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 7, 1> f_vector;
    double f_1 = sqrt(odometry_weights_.at(0)) * (vars[0] - 1);
    double f_2 = sqrt(odometry_weights_.at(1)) * (vars[1] - vars[0] - 0.25);
    double f_3 = sqrt(odometry_weights_.at(2)) * (vars[2] - vars[1] - 0.25);
    double f_4 = sqrt(odometry_weights_.at(3)) * (vars[3] - vars[2] - 0.25);
    double f_5 = sqrt(odometry_weights_.at(4)) * (vars[4] - vars[3] - 0.25);
    double f_6 = sqrt(odometry_weights_.at(5)) * (vars[5] - vars[4] - 1);
    double f_7 = sqrt(loop_closure_weight_) * (vars[5] - loop_closures_length_);
    f_vector << f_1, f_2, f_3, f_4, f_5, f_6, f_7;
    // std::cout << "f_vector: " << f_vector << std::endl;
    return f_vector;
  }

private:
  std::vector<double> odometry_weights_;

  // >= 4
  double loop_closures_length_;
  double loop_closure_weight_;
};

class SpringJacobianSmallerConstraints : public Jacobian {
public:
  SpringJacobianSmallerConstraints(const std::vector<double> &odometry_weights,
                                   const double loop_closure_weight)
      : odometry_weights_(odometry_weights),
        loop_closure_weight_(loop_closure_weight) {}

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  operator()(const double *vars) const override {
    Eigen::Matrix<double, 7, 6> jacobian;
    jacobian.row(0) << sqrt(odometry_weights_.at(0)), 0.0, 0.0, 0.0, 0.0, 0.0;
    jacobian.row(1) << (-1.0) * sqrt(odometry_weights_.at(1)),
        sqrt(odometry_weights_.at(1)), 0.0, 0.0, 0.0, 0.0;
    jacobian.row(2) << 0.0, (-1.0) * sqrt(odometry_weights_.at(2)),
        sqrt(odometry_weights_.at(2)), 0.0, 0.0, 0.0;
    jacobian.row(3) << 0.0, 0.0, (-1.0) * sqrt(odometry_weights_.at(3)),
        sqrt(odometry_weights_.at(3)), 0.0, 0.0;
    jacobian.row(4) << 0.0, 0.0, 0.0, (-1.0) * sqrt(odometry_weights_.at(4)),
        sqrt(odometry_weights_.at(4)), 0.0;
    jacobian.row(5) << 0.0, 0.0, 0.0, 0.0,
        (-1.0) * sqrt(odometry_weights_.at(5)), sqrt(odometry_weights_.at(5));
    jacobian.row(6) << 0.0, 0.0, 0.0, 0.0, 0.0, sqrt(loop_closure_weight_);
    // std::cout << "jacobian: " << jacobian << std::endl;

    return jacobian;
  }

private:
  std::vector<double> odometry_weights_;
  double loop_closure_weight_;
};

} // namespace optimisation