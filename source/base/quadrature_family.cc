#include <fiddle/base/quadrature_family.h>
#include <fiddle/base/utilities.h>

#include <deal.II/base/quadrature_lib.h>

#include <utility>

namespace fdl
{
  template <int dim>
  QGaussFamily<dim>::QGaussFamily(const unsigned int min_points_1D,
                                  const double       point_density)
    : min_points_1D(min_points_1D)
    , point_density(point_density)
  {}

  template <int dim>
  unsigned char
  QGaussFamily<dim>::get_n_points_1D(const double eulerian_length,
                                     const double lagrangian_length) const
  {
    const double n_evenly_spaced_points =
      std::ceil(point_density * lagrangian_length / eulerian_length);
    const double min_point_distance = 1.0 / n_evenly_spaced_points;

    // TODO: use binary search instead if max_point_distances.back() <
    // min_point_distance
    unsigned char i = 0;
    while (true)
      {
        // access the quadrature first to guarantee that max_point_distances is
        // long enough
        this->operator[](i);
        Assert(i < max_point_distances.size(), ExcFDLInternalError());
        const double max_distance = max_point_distances[i];

        if (max_distance <= min_point_distance)
          return i;

        if (i == std::numeric_limits<unsigned char>::max())
          break;
        else
          ++i;
      }

    Assert(false, ExcFDLInternalError());

    return std::numeric_limits<unsigned char>::max();
  }

  template <int dim>
  const Quadrature<dim> &
  QGaussFamily<dim>::operator[](const unsigned char n_points_1D) const
  {
    if (n_points_1D < quadratures.size())
      {
        return quadratures[n_points_1D];
      }
    else
      {
        // compute all the remaining quadratures up to that point so that
        // the deque doesn't have holes
        for (unsigned char index = quadratures.size(); index <= n_points_1D;
             ++index)
          {
            // This is a bit tricky since we can either iterate the
            // lower-order rule or increase the number of quadrature points
            // to achieve a well-spaced quadrature rule. Try a few and go
            // with whichever uses the smallest number of points.
            const auto n_tries = std::min<std::size_t>(16u, n_points_1D + 2);
            std::vector<std::pair<unsigned int, unsigned int>> pairs(n_tries);
            for (unsigned int i = 0; i < n_tries; ++i)
              {
                // we always need at least one iteration
                const auto n_iterations =
                  std::max(1u,
                           static_cast<unsigned int>(
                             std::ceil(double(index) / (min_points_1D + i))));

                pairs[i] = std::make_pair(min_points_1D + i, n_iterations);
              }
            std::sort(pairs.begin(),
                      pairs.end(),
                      [](const std::pair<unsigned int, unsigned int> &a,
                         const std::pair<unsigned int, unsigned int> &b) {
                        const auto p1 = a.first * a.second;
                        const auto p2 = b.first * b.second;
                        // prefer rules with fewer (1D) points:
                        if (p1 != p2)
                          {
                            return p1 < p2;
                          }
                        // in case we have a tie, prefer the rule with the
                        // lower order (it likely has better spacing)
                        else
                          {
                            return a.first < b.first;
                          }
                      });

            QIterated<dim> new_quad(QGauss<1>(pairs[0].first), pairs[0].second);
            // its OK to round here since we have an integer root (up to
            // roundoff)
            Assert(index <= static_cast<unsigned char>(
                              std::round(std::pow(new_quad.size(), 1.0 / dim))),
                   ExcFDLInternalError());
            Assert(quadratures.size() == std::size_t(index),
                   ExcFDLInternalError());
            quadratures.emplace_back(new_quad);
            // If we only have one point it should be in the center - just set
            // the diameter to 1
            if (new_quad.get_points().size() == 1)
              max_point_distances.push_back(1.0);
            else
              max_point_distances.push_back(
                find_largest_nonintersecting_sphere(new_quad.get_points())
                  .second);
          }

        Assert(n_points_1D < quadratures.size(), ExcFDLInternalError());
        Assert(max_point_distances.size() == quadratures.size(),
               ExcFDLInternalError());
        return quadratures[n_points_1D];
      }
  }

  template class QGaussFamily<NDIM - 1>;
  template class QGaussFamily<NDIM>;
} // namespace fdl
