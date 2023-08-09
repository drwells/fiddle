#ifndef included_fiddle_utilities_h
#define included_fiddle_utilities_h

#include <fiddle/base/config.h>

#include <deal.II/base/point.h>

#include <string>
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
    const std::vector<Point<spacedim>> &points);

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
