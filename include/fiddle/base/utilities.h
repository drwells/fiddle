#ifndef included_fiddle_utilities_h
#define included_fiddle_utilities_h

#include <deal.II/base/point.h>

#include <utility>

namespace fdl
{
  using namespace dealii;

  /**
   * Given a set of points, find the largest sphere which is tangent to two
   * points, does not contain any points inside the set, and whose center is
   * the average of two points. The diameter of that sphere is a good measure
   * of the maximum distance of any point from its neighbors.
   */
  template <int spacedim>
  std::pair<Point<spacedim>, double>
  find_largest_nonintersecting_sphere(
    const std::vector<Point<spacedim>> &points)
  {
    Assert(points.size() > 1,
           ExcMessage("Need at least two points to compute a distance"));
    Point<spacedim> best_center;
    double          best_diameter = 0.0;

    // This is the brute-force version. We don't normally have more than a
    // hundred quadrature points per cell, so N^2 algorithms aren't
    // completely deadly yet.
    auto sphere_contains_nontangent_point =
      [&points](const Point<spacedim> &center,
                const Point<spacedim> &tangent_point_1,
                const Point<spacedim> &tangent_point_2,
                const double           diameter) -> bool {
      const double magnitude = std::max(center.norm(), tangent_point_1.norm());
      for (const auto &point : points)
        if ((point != tangent_point_1) && (point != tangent_point_2) &&
            // relax the check a little bit by ensuring that the point
            // is actually inside the sphere with some tolerance
            (point.distance(center) - diameter / 2.0) < -magnitude * 1e-14)
          return true;
      return false;
    };

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto &point1 = points[i];
        for (unsigned int j = i + 1; j < points.size(); ++j)
          {
            const auto &          point2             = points[j];
            const Point<spacedim> tentative_center   = (point1 + point2) / 2.0;
            const double          tentative_diameter = point1.distance(point2);

            if ((tentative_diameter > best_diameter) &&
                !sphere_contains_nontangent_point(
                  tentative_center, point1, point2, tentative_diameter))
              {
                best_center   = tentative_center;
                best_diameter = tentative_diameter;
              }
          }
      }

    return std::make_pair(best_center, best_diameter);
  }
} // namespace fdl

#endif
