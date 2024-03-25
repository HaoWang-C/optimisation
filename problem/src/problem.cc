#include "problem.h"

namespace optimisation {

Eigen::Matrix<double, 2, 1> ExampleProblem::func() const {
  Eigen::Matrix<double, 2, 1> f;
  f << (x_ + 1), (x_ * x_ + x_ - 1);
  return f;
}

double ExampleProblem::CostFunc() const {
  Eigen::Matrix<double, 2, 1> f_matrix = func();
  return 0.5 * (f_matrix(0) * f_matrix(0) + f_matrix(1) * f_matrix(1));
}

Eigen::Matrix<double, 2, 1> ExampleProblem::Jacobian_of_f() const {
  Eigen::Matrix<double, 2, 1> Jacobian;
  Jacobian << 1, (2 * x_ + 1);
  return Jacobian;
}

bool ExampleProblem::UpdateVariable(const std::vector<double> &delta_x) {
  if (delta_x.size() != 1) {
    return false;
  }
  x_ += delta_x.at(0);
  std::cout << "Update variable to: " << x_ << std::endl;
  return true;
}

} // namespace optimisation