#ifndef included_fiddle_base_quadrature_family_h
#define included_fiddle_base_quadrature_family_h

#include <deal.II/base/quadrature.h>

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
    virtual const Quadrature<dim> &
    operator[](const unsigned char n_points_1D) const = 0;
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
    operator[](const unsigned char n_points_1D) const
    {
      return single_quad;
    }

  protected:
    Quadrature<dim> single_quad;
  };
} // namespace fdl

#endif
