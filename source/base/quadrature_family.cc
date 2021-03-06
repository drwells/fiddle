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
  QGaussFamily<dim>::get_index(const double eulerian_length,
                               const double lagrangian_length) const
  {
    // It isn't really a number of points, but we get more accurate answers if
    // we don't round or use ceil here.
    const double n_evenly_spaced_points =
      point_density * lagrangian_length / eulerian_length;
    const double min_point_distance = 1.0 / n_evenly_spaced_points;

    // TODO: use binary search instead if mean_point_distances.back() <
    // min_point_distance
    unsigned char i = 0;
    while (true)
      {
        // access the quadrature first to guarantee that mean_point_distances is
        // long enough
        this->operator[](i);
        Assert(i < mean_point_distances.size(), ExcFDLInternalError());
        const double mean_distance = mean_point_distances[i];

        if (mean_distance <= min_point_distance)
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
        const double shrink_factor = 0.05;
        const double target_point_distance =
          max_point_distances.size() == 0 ?
            1.0 :
            (1.0 - shrink_factor) * max_point_distances.back();
        // compute all the remaining quadratures up to that point so that
        // the deque doesn't have holes
        for (unsigned char index = quadratures.size(); index <= n_points_1D;
             ++index)
          {
            // This is a bit tricky since we can either iterate the
            // lower-order rule or increase the number of quadrature points
            // to achieve a well-spaced quadrature rule. Try a few and go
            // with whichever uses the smallest number of points.
            const auto n_tries = std::min<std::size_t>(16u, n_points_1D + 6);
            std::vector<std::pair<unsigned int, unsigned int>> pairs;
            // Try 1: increase order
            for (unsigned int i = 0; i < n_tries; ++i)
              {
                // we always need at least one iteration
                const auto n_iterations = std::max<unsigned int>(
                  1u, std::ceil(double(index) / (min_points_1D + i)));

                pairs.emplace_back(min_points_1D + i, n_iterations);
              }

            // Try 2: use more iterations. min_points_1D is pretty low so we can
            // assume we are using roughly evenly spaced points
            const auto ratio0 = std::max<unsigned int>(
              1, std::ceil(double(n_points_1D) / min_points_1D));
            const auto ratio1 = std::max<unsigned int>(
              1, std::ceil(double(n_points_1D + 1) / min_points_1D));
            const auto ratio2 = std::max<unsigned int>(
              1, std::ceil(double(n_points_1D + 2) / min_points_1D));
            // This can get expensive so don't try that many at higher order
            for (int i = 0; i < std::max<int>(6, 10 - min_points_1D); ++i)
              {
                pairs.emplace_back(min_points_1D, ratio0 + i);
                pairs.emplace_back(min_points_1D + 1, ratio1 + i);
                pairs.emplace_back(min_points_1D + 2, ratio2 + i);
              }
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

            // Getting good quadrature rules really pays off so put a lot of
            // effort into sorting these guys and picking the one with the
            // fewest number of points which meets the spacing criterion
            std::size_t best_index    = 0;
            std::size_t best_n_points = std::numeric_limits<std::size_t>::max();
            double      best_point_distance = 1.0;
            for (std::size_t i = 0; i < pairs.size(); ++i)
              {
                // We don't have end points in these quadratures so this formula
                // always works
                const std::size_t n_points =
                  std::pow(pairs[i].first * pairs[i].second, dim);

                if (n_points <= best_n_points)
                  {
                    const QIterated<dim> new_quad(QGauss<1>(pairs[i].first),
                                                  pairs[i].second);
                    Assert(new_quad.size() == n_points, ExcFDLInternalError());
                    const double point_distance =
                      new_quad.size() < 2 ? 1.0 :
                                            find_largest_nonintersecting_sphere(
                                              new_quad.get_points())
                                              .second;

                    // If we have the same number of points, pick the rule with
                    // better spacing
                    if (n_points == best_n_points &&
                        point_distance < best_point_distance)
                      {
                        best_index          = i;
                        best_n_points       = new_quad.size();
                        best_point_distance = point_distance;
                      }

                    // If we have different numbers of points, pick the rule
                    // with fewer points
                    if (n_points < best_n_points &&
                        point_distance < target_point_distance)
                      {
                        best_index          = i;
                        best_n_points       = new_quad.size();
                        best_point_distance = point_distance;
                      }
                  }
              }
            Assert(best_n_points != std::numeric_limits<std::size_t>::max(),
                   ExcFDLInternalError());
            QIterated<dim> new_quad(QGauss<1>(pairs[best_index].first),
                                    pairs[best_index].second);
            // its OK to round here since we have an integer root (up to
            // roundoff)
            Assert(index <= static_cast<unsigned char>(
                              std::round(std::pow(new_quad.size(), 1.0 / dim))),
                   ExcFDLInternalError());
            Assert(quadratures.size() == std::size_t(index),
                   ExcFDLInternalError());
            quadratures.emplace_back(std::move(new_quad));

            max_point_distances.emplace_back(best_point_distance);
            const unsigned int n_1D_points =
              pairs[best_index].first * pairs[best_index].second;
            mean_point_distances.emplace_back(1.0 / n_1D_points);
          }

        Assert(n_points_1D < quadratures.size(), ExcFDLInternalError());
        Assert(max_point_distances.size() == quadratures.size(),
               ExcFDLInternalError());
        Assert(mean_point_distances.size() == quadratures.size(),
               ExcFDLInternalError());
        return quadratures[n_points_1D];
      }
  }

  // It would probably be better to add a base class instead of copying and
  // pasting
  template <int dim>
  QWitherdenVincentSimplexFamily<dim>::QWitherdenVincentSimplexFamily(
    const unsigned int min_points_1D,
    const double       point_density)
    : min_points_1D(min_points_1D)
    , point_density(point_density)
  {}

  template <int dim>
  unsigned char
  QWitherdenVincentSimplexFamily<dim>::get_index(
    const double eulerian_length,
    const double lagrangian_length) const
  {
    // It isn't really a number of points, but we get more accurate answers if
    // we don't round or use ceil here.
    const double n_evenly_spaced_points =
      point_density * lagrangian_length / eulerian_length;
    const double min_point_distance = 1.0 / n_evenly_spaced_points;

    // TODO: use binary search instead if mean_point_distances.back() <
    // min_point_distance
    unsigned char i = 0;
    while (true)
      {
        // access the quadrature first to guarantee that mean_point_distances is
        // long enough
        this->operator[](i);
        Assert(i < mean_point_distances.size(), ExcFDLInternalError());
        const double mean_distance = mean_point_distances[i];

        if (mean_distance <= min_point_distance)
          return i;

        if (i == std::numeric_limits<unsigned char>::max())
          break;
        else
          ++i;
      }

    Assert(false, ExcFDLInternalError());

    return std::numeric_limits<unsigned char>::max();
  }

  // Similarly, this is more-or-less copy and paste code but with a few key
  // things changed (we can only do powers of 2 for iterated simplex rules)
  template <int dim>
  const Quadrature<dim> &
  QWitherdenVincentSimplexFamily<dim>::operator[](
    const unsigned char n_points_1D) const
  {
    if (n_points_1D < quadratures.size())
      {
        return quadratures[n_points_1D];
      }
    else
      {
        Assert(dim == 2 || dim == 3, ExcNotImplemented());
        const double shrink_factor = 0.05;
        const double target_point_distance =
          max_point_distances.size() == 0 ?
            1.0 :
            (1.0 - shrink_factor) * max_point_distances.back();
        // compute all the remaining quadratures up to that point so that
        // the deque doesn't have holes
        for (unsigned char index = quadratures.size(); index <= n_points_1D;
             ++index)
          {
            auto next_power_of_2 = [](const unsigned int lower_bound) {
              unsigned int result = 1;
              while (result < lower_bound)
                result = result << 1;
              return result;
            };

            // This is a bit tricky since we can either iterate the
            // lower-order rule or increase the number of quadrature points
            // to achieve a well-spaced quadrature rule. Try a few and go
            // with whichever uses the smallest number of points.
            const auto n_tries = std::min<std::size_t>(20u, n_points_1D + 10);
            // QWVS is missing higher-order rules
            const unsigned int max_n_points_1D = dim == 2 ? 6 : 5;
            Assert(min_points_1D <= max_n_points_1D, ExcFDLNotImplemented());
            std::vector<std::pair<unsigned int, unsigned int>> pairs;
            // Try 1: increase order.
            for (unsigned int i = 0; i < n_tries; ++i)
              {
                // we always need at least one iteration
                const auto n_target_iterations = std::max<unsigned int>(
                  1u, std::ceil(double(index) / (min_points_1D + i)));
                pairs.emplace_back(std::min(max_n_points_1D, min_points_1D + i),
                                   next_power_of_2(n_target_iterations));
              }

            // Try 2: use more iterations.
            for (int i = 0; i < 6; ++i)
              {
                pairs.emplace_back(min_points_1D, std::pow(2, i));
                pairs.emplace_back(std::min(max_n_points_1D, min_points_1D + 1),
                                   std::pow(2, i));
                pairs.emplace_back(std::min(max_n_points_1D, min_points_1D + 2),
                                   std::pow(2, i));
                pairs.emplace_back(std::min(max_n_points_1D, min_points_1D + 3),
                                   std::pow(2, i));
                pairs.emplace_back(std::min(max_n_points_1D, min_points_1D + 4),
                                   std::pow(2, i));
              }

            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
            // Try to avoid really huge quadrature rules if we can
            std::stable_sort(
              pairs.begin(),
              pairs.end(),
              [](const std::pair<unsigned int, unsigned int> &a,
                 const std::pair<unsigned int, unsigned int> &b) {
                return a.first * a.second < b.first * b.second;
              });
            // TODO - compute sizes in some other way
            std::vector<unsigned int> points_per_q_rule;
            if (dim == 2)
              points_per_q_rule = {0, 1, 6, 7, 15, 19, 28};
            else
              points_per_q_rule = {0, 1, 8, 14, 35, 59};
            // Getting good quadrature rules really pays off so put a lot of
            // effort into sorting these guys and picking the one with the
            // fewest number of points which meets the spacing criterion
            std::size_t best_index    = 0;
            std::size_t best_n_points = std::numeric_limits<std::size_t>::max();
            double      best_point_distance = 1.0;
            for (std::size_t i = 0; i < pairs.size(); ++i)
              {
                const unsigned int n_base_points =
                  points_per_q_rule[pairs[i].first];
                const auto power = static_cast<unsigned int>(
                  std::round(std::log2(pairs[i].second)));
                const auto n_copies = static_cast<unsigned int>(
                  std::round(std::pow(std::pow(2, dim), power)));
                const unsigned int n_points = n_base_points * n_copies;

                if (n_points <= best_n_points)
                  {
                    const QIteratedSimplex<dim> new_quad(
                      QWitherdenVincentSimplex<dim>(pairs[i].first),
                      pairs[i].second);
                    Assert(new_quad.size() == n_points, ExcFDLInternalError());

                    const double point_distance =
                      new_quad.size() < 2 ? 1.0 :
                                            find_largest_nonintersecting_sphere(
                                              new_quad.get_points())
                                              .second;

                    // If we have the same number of points, pick the rule with
                    // better spacing
                    if (n_points == best_n_points &&
                        point_distance < best_point_distance)
                      {
                        best_index          = i;
                        best_n_points       = new_quad.size();
                        best_point_distance = point_distance;
                      }

                    // If we have different numbers of points, pick the rule
                    // with fewer points
                    if (n_points < best_n_points &&
                        point_distance < target_point_distance)
                      {
                        best_index          = i;
                        best_n_points       = new_quad.size();
                        best_point_distance = point_distance;
                      }
                  }
              }
            Assert(best_n_points != std::numeric_limits<std::size_t>::max(),
                   ExcFDLInternalError());
            QIteratedSimplex<dim> new_quad(QWitherdenVincentSimplex<dim>(
                                             pairs[best_index].first),
                                           pairs[best_index].second);
            Assert(quadratures.size() == std::size_t(index),
                   ExcFDLInternalError());
            // Get a 1D estimate by counting the number of points that would be
            // in the equivalent composite rule in a hypercube
            const auto n_copies =
              static_cast<unsigned int>(std::round(std::tgamma(dim + 1)));
            // As usual, there is no need to round here
            const auto n_1D_points =
              std::pow(new_quad.size() * n_copies, 1.0 / dim);
            mean_point_distances.emplace_back(1.0 / n_1D_points);
            max_point_distances.emplace_back(best_point_distance);
            quadratures.emplace_back(std::move(new_quad));
            Assert(index <= static_cast<unsigned char>(n_1D_points),
                   ExcFDLInternalError());
          }

        Assert(n_points_1D < quadratures.size(), ExcFDLInternalError());
        Assert(max_point_distances.size() == quadratures.size(),
               ExcFDLInternalError());
        Assert(mean_point_distances.size() == quadratures.size(),
               ExcFDLInternalError());
        return quadratures[n_points_1D];
      }
  }

  template class QGaussFamily<NDIM - 1>;
  template class QGaussFamily<NDIM>;

  template class QWitherdenVincentSimplexFamily<NDIM - 1>;
  template class QWitherdenVincentSimplexFamily<NDIM>;
} // namespace fdl
