#include "problem.h"

namespace optimisation {

void printArray(const double *array, int size) {
  std::cout << "[";
  for (int i = 0; i < size; ++i) {
    std::cout << array[i];
    if (i < size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]";
}

Eigen::Matrix<double, Eigen::Dynamic, 1>
LeastSquareProblem::EvaluateVectorFunction() const {
  return (*func_)(x_);
}

Eigen::Matrix<double, Eigen::Dynamic, 1>
LeastSquareProblem::EvaluateVectorFunctionAtGivenValue(const double *x) const {
  return (*func_)(x);
}

double
LeastSquareProblem::EvaluateCostFunctionAtGivenValue(const double *x) const {
  Eigen::Matrix<double, Eigen::Dynamic, 1> f_vector =
      EvaluateVectorFunctionAtGivenValue(x);
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