#pragma once

#include "problem.h"
#include <iostream>

namespace unconstrained {
// The following parameters are need for LM algorithm:
//  1. number_of_iteration: max iteration time
//  2. damping_parameter: damping parameter for increasing or decreasing the
//  step
//  3. damping_factor: factor used for scale up the damping parameter is the
//  gain ratio is negative
//  4. gradient_criteria: stopping criteria if gradient change is small || g ||
//  <= gradient_criteria
//  5. parameter_criteria: stopping criteria if parameter chang is small || x ||
//  <= parameter_criteria
class LevenBergMarquardtSolver {
public:
  LevenBergMarquardtSolver(optimisation::LeastSquareProblem &problem,
                           const int numer_of_iteration,
                           const int damping_factor,
                           const int gradient_criteria,
                           const int parameter_criteria)
      : problem_(problem), numer_of_iteration_(numer_of_iteration),
        damping_factor_(damping_factor), gradient_criteria_(gradient_criteria_),
        parameter_criteria_(parameter_criteria) {}

  bool Solve();

private:
  // Evaluate A = J^t * J
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EvaluateMatrixA() const;

  // Evaluate g = J^t * f
  Eigen::Matrix<double, Eigen::Dynamic, 1> EvaluateVectorG() const;

  // Initialise damping_parameter
  void InitialiseDampingParameters();

  // Stopping criteria
  bool CheckGradientCriteria() const;
  bool CheckParametersCriteria() const;

  // Calculate h_lm by solving (A + miu * I) * h_lm = -g
  Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateLMStep() const;

  // gain ratio is calculated by (F(x) - F(x_new)) / (L(0) - L(h_lm))
  double CalculateGainRatio() const;

  // Depending on the gain ratio
  void UpdateDampingParameter(const double gain_ratio);

  optimisation::LeastSquareProblem &problem_;
  int numer_of_iteration_;
  // Initialised by the algorithm
  // TODO: optionally can be specifed by user
  double damping_parameter_;
  // Given by user
  double damping_factor_;
  double gradient_criteria_;
  double parameter_criteria_;
  // If a solutions is found
  bool is_converge{false};
};

} // namespace unconstrained
