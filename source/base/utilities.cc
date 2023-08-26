#include <fiddle/base/exceptions.h>
#include <fiddle/base/utilities.h>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include <array>

namespace fdl
{
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
    auto x_less = [](const Point<spacedim> &a, const Point<spacedim> &b)
    { return a[0] < b[0]; };

    std::vector<Point<spacedim>> points_copy(points);
    std::sort(points_copy.begin(), points_copy.end(), x_less);

    // This is the brute-force version. We don't normally have more than a
    // hundred quadrature points per cell, so N^2 algorithms aren't
    // completely deadly yet.
    auto sphere_contains_nontangent_point =
      [&](const Point<spacedim> &center,
          const unsigned int    &tangent_point_n,
          const double           diameter) -> bool
    {
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
  std::string
  encode_base64(const char *begin, const char *end)
  {
    using namespace boost::archive::iterators;
    using iterator = base64_from_binary<transform_width<const char *, 6, 8>>;
    std::string base64{iterator(begin), iterator(end)};
    // Add padding.
    std::array<std::string, 3> paddings{"", "==", "="};
    base64.append(paddings[(end - begin) % 3]);

    return base64;
  }

  std::string
  decode_base64(const char *begin, const char *end)
  {
    using namespace boost::archive::iterators;
    using iterator = transform_width<binary_from_base64<const char *>, 8, 6>;
    std::string binary{iterator(begin), iterator(end)};

    // We have three possibilities for padding, based on how boost decodes it:
    // input ends in "==": remove two NULs at the end
    // input ends in "=": remove one NUL at the end
    // otherwise: no padding, nothing to remove
    if (begin != end)
      {
        const char *input = end - 1;
        while (input >= begin && *input == '=')
          {
            binary.pop_back();
            --input;
          }
      }
    return binary;
  }

  //
  // instantiations
  //

  template std::pair<Point<1>, double>
  compute_largest_nonintersecting_sphere(const std::vector<Point<1>> &);

  template std::pair<Point<2>, double>
  compute_largest_nonintersecting_sphere(const std::vector<Point<2>> &);

  template std::pair<Point<3>, double>
  compute_largest_nonintersecting_sphere(const std::vector<Point<3>> &);
} // namespace fdl
