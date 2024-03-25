#pragma once

#include "problem.h"
#include <iostream>

namespace unconstrained {
void GaussianNewtonSolve(optimisation::LeastSquareProblem &problem,
                         const int numer_of_iteration);
} // namespace unconstrained
