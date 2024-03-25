#include "problem.h"

namespace optimisation {

Eigen::Matrix<double, Eigen::Dynamic, 1>
LeastSquareProblem::EvaluateVectorFunction() const {
  return (*func_)(x_);
}

double LeastSquareProblem::EvaluateCostFunction() const {
  Eigen::Matrix<double, Eigen::Dynamic, 1> f_vector = EvaluateVectorFunction();
  return 0.5 * f_vector.squaredNorm();
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
LeastSquareProblem::EvaluateJacobian() const {
  return (*jacobian_)(x_);
}

void LeastSquareProblem::UpdateVariable(
    const Eigen::Matrix<double, Eigen::Dynamic, 1> &h) {
  // Check if the sizes of x_ and h match
  if (h.size() != size_of_x_) {
    // TODO: throw exception
    std::cout << "The step h has wrong size" << std::endl;
  }

  // Perform element-wise addition
  for (int i = 0; i < h.size(); ++i) {
    x_[i] += h(i);
  }
}

} // namespace optimisation