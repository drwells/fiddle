#ifndef included_fiddle_base_quadrature_family_h
#define included_fiddle_base_quadrature_family_h

#include <fiddle/base/exceptions.h>

#include <deal.II/base/quadrature.h>

#include <deque>

namespace fdl
{
  using namespace dealii;

  /**
   * Implementation of a family of quadratures - meaning that, for a specified
   * number of points in a single coordinate direction, objects of this class
   * return a constant reference to a quadrature meeting that requirement.
   */
  template <int dim>
  class QuadratureFamily : public Subscriptor
  {
  public:
    /**
     * Return a quadrature rule that uses at least @p n_points_1D quadrature
     * points in each coordinate direction.
     */
    virtual const Quadrature<dim> &
    operator[](const unsigned char n_points_1D) const = 0;

    /**
     * Determine how many 1D quadrature points we need to satisfy the given
     * density condition.
     *
     * @param[in] eulerian_length Length of one of the edges of an Eulerian cell.
     *
     * @param[in] lagrangian_length Length scale (typically the length of the
     * longest edge of an element bounding box) of a Lagrangian cell.
     */
    virtual unsigned char
    get_n_points_1D(const double eulerian_length,
                    const double lagrangian_length) const = 0;
  };

  /**
   * Family of quadratures represented by a single rule, regardless of the
   * number of 1D quadrature points. The primary utility of this class is
   * testing.
   */
  template <int dim>
  class SingleQuadrature : public QuadratureFamily<dim>
  {
  public:
    SingleQuadrature(const Quadrature<dim> &quad)
      : single_quad(quad)
    {}

    virtual const Quadrature<dim> &
    operator[](const unsigned char /*n_points_1D*/) const override
    {
      return single_quad;
    }

    virtual unsigned char
    get_n_points_1D(const double /*eulerian_length*/,
                    const double /*lagrangian_length*/) const override
    {
      std::size_t n_points_1D = 1;
      while (std::pow(n_points_1D, dim) < single_quad.size())
        ++n_points_1D;

      Assert(n_points_1D < std::size_t(
               std::numeric_limits<unsigned char>::max()),
             ExcFDLNotImplemented());
      return static_cast<unsigned char>(n_points_1D);
    }

  protected:
    Quadrature<dim> single_quad;
  };

  /**
   * Family of quadratures satisfying some minimum order condition based on
   * QGauss and QIterated.
   *
   * This class will alternate between increasing the number of iterated
   * quadratures and the order of the method to generate rules with the fewest
   * number of points that still satisfy the minimum order condition.
   */
  template <int dim>
  class QGaussFamily : public QuadratureFamily<dim>
  {
    public:
      /**
       * Constructor. Takes, as argument, the minimum number of 1D points that
       * generated quadrature rules will have. Here, like for
       * dealii::Quadrature, @p min_points_1D should be understood as a
       * parameter specifying the order of the quadratures.
       */
      QGaussFamily(const unsigned int min_points_1D, const double point_density = 1.0);

      virtual const Quadrature<dim> &
      operator[](const unsigned char n_points_1D) const override;

    virtual unsigned char
    get_n_points_1D(const double eulerian_length,
                    const double lagrangian_length) const override;

    protected:
      unsigned int min_points_1D;

      double point_density;

      void add_new_rule(const unsigned char);

      /**
       * Quadratures. Left as mutable so that new quadratures can be added in
       * operator[] should the need arise. Uses a deque so that references are
       * not invalidated.
       */
      mutable std::deque<Quadrature<dim>> quadratures;

    /**
     * Maximum distance between nearest neighbors of quadrature points. Cached
     * here to make get_n_points_1D a lot faster.
     */
    mutable std::vector<double> max_point_distances;
  };
} // namespace fdl

#endif
