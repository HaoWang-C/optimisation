#pragma once
// Design a optimisation problem then design a sovler. This code design the problem.
// We need:
//    1. A cost function F that can be evaluated at x
//    2. The Jacobian that can be evaluated at x
//    3. The Hessian that can be evaluated at x

// Design:
//  1. The problem shoud be a "class".
//  2. The problem class should own F J H as functor.
//  3. The problem should also be able to access/update the optimisation variable
//  4. The optimisation variable x should be a double value vector

namespace optimisation
{

} // namespace unconstrained