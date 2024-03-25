#pragma once

#include "problem.h"
#include <iostream>

namespace unconstrained {
template <int m, int n>
void GaussianNewtonSolve(optimisation::LeastSquareProblem<m, n> &problem);

// Template specialization for m = 2 and n = 1 for testing
template <>
void GaussianNewtonSolve<2, 1>(optimisation::LeastSquareProblem<2, 1> &problem);

} // namespace unconstrained
