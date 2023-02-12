#ifndef included_fiddle_utilities_h
#define included_fiddle_utilities_h

#include <fiddle/base/config.h>

#include <deal.II/base/point.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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
  compute_largest_nonintersecting_sphere(
    const std::vector<Point<spacedim>> &points)
  {
    Assert(points.size() > 1,
           ExcMessage("Need at least two points to compute a distance"));
    Point<spacedim> best_center;
    double          best_diameter = 0.0;

    // Speed up the bounding check by only examining points whose x coordinate
    // is near that of the sphere
    auto x_less = [](const Point<spacedim> &a, const Point<spacedim> &b) {
      return a[0] < b[0];
    };

    std::vector<Point<spacedim>> points_copy(points);
    std::sort(points_copy.begin(), points_copy.end(), x_less);

    // This is the brute-force version. We don't normally have more than a
    // hundred quadrature points per cell, so N^2 algorithms aren't
    // completely deadly yet.
    auto sphere_contains_nontangent_point =
      [&](const Point<spacedim> &center,
          const unsigned int    &tangent_point_n,
          const double           diameter) -> bool {
      const double magnitude =
        std::max(center.norm(), points[tangent_point_n].norm());

      Point<spacedim> left  = center;
      Point<spacedim> right = center;
      left[0] -= diameter / 2.0;
      right[0] += diameter / 2.0;

      // Only check points that are in the correct x pencil.
      auto upper_bound =
        std::lower_bound(points_copy.begin(), points_copy.end(), right, x_less);
      auto lower_bound =
        std::lower_bound(points_copy.begin(), points_copy.end(), left, x_less);
      if (lower_bound != points_copy.begin())
        --lower_bound;

      for (auto it = lower_bound; it < upper_bound; ++it)
        if ((it->distance(center) - diameter / 2.0) < -magnitude * 1e-14)
          return true;

      return false;
    };

    // points tend to be clustered at endpoints. Try to find one in the middle
    // to start with:
    {
      const unsigned int i      = points.size() / 2;
      const auto        &point1 = points[i];
      for (unsigned int j = i + 1; j < points.size(); ++j)
        {
          const auto           &point2             = points[j];
          const Point<spacedim> tentative_center   = (point1 + point2) / 2.0;
          const double          tentative_diameter = point1.distance(point2);

          if ((tentative_diameter > best_diameter) &&
              !sphere_contains_nontangent_point(tentative_center,
                                                i,
                                                tentative_diameter))
            {
              best_center   = tentative_center;
              best_diameter = tentative_diameter;
            }
        }
    }

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto &point1 = points[i];
        for (unsigned int j = i + 1; j < points.size(); ++j)
          {
            const auto           &point2             = points[j];
            const Point<spacedim> tentative_center   = (point1 + point2) / 2.0;
            const double          tentative_diameter = point1.distance(point2);

            if ((tentative_diameter > best_diameter) &&
                !sphere_contains_nontangent_point(tentative_center,
                                                  i,
                                                  tentative_diameter))
              {
                best_center   = tentative_center;
                best_diameter = tentative_diameter;
              }
          }
      }

    return std::make_pair(best_center, best_diameter);
  }

  /**
   * Encode arbitrary data from binary to base64. The output type is a string so
   * that this can be easily saved to a SAMRAI database.
   */
  std::string
  encode_base64(const char *begin, const char *end);

  /**
   * Decode arbitrary data from base64 back to binary. The output type is a
   * string so that this can be easily read from with std::istringstream.
   */
  std::string
  decode_base64(const char *begin, const char *end);
} // namespace fdl

#endif
