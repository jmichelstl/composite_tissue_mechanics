 /* This code generates points according to random point processes. Two methods
  * are offered. In the first case, a Poisson sphere packing is computed, in
  * which points are distributed as densely as possible such that no two points
  * are closer than a specified minimum distance. In the second case, a 
  * hyperuniform set of points, which has a two-point correlation function that
  * decays more slowly than that of a random point packing but which has no 
  * long-range orientational order. Given a bounding  box and a specified 
  * density of vertices, points are initially placed according to a Poisson 
  * sphere packing, then evolved to a hyperuniform state according to Lloyd 
  * moves. 
  */

#ifndef RPP
#define RPP

#include <vector>
#include "geometric_primitives.hpp"

//Empirical conversion factor to go from a desired density to a minimum distance
//in a Poisson sphere packing
#define DENSITY_TO_LENGTH 1.077217345

std::vector<double> poisson_sphere_packing(Bounds3D bounds, double cutoff);

std::vector<double> compute_hyperuniform_packing(Bounds3D bounds, double density, double conv_tol, int max_iter);

#endif
