#ifndef included_fiddle_mechanics_force_contribution_h
#define included_fiddle_mechanics_force_contribution_h

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_update_flags.h>

#include <fiddle/mechanics/mechanics_values.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Interface class for force contributions from various sources to Parts.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class ForceContribution
  {
  public:
    /**
     * Constructor
     */
    ForceContribution(const Quadrature<dim> &quad)
      : quadrature(quad)
    {}

    const Quadrature<dim> &
    get_quadrature() const
    {
      return quadrature;
    }

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsValuesFlags
    get_mechanics_values_flags() const = 0;

    /**
     * Get the update flags this force contribution requires for FEValues
     * objects.
     */
    virtual UpdateFlags
    get_update_flags() const = 0;


    /**
     * Compute forces at quadrature points. Should work regardless of whether
     * we are on the surface of the element or inside it.
     */
    virtual void
    compute_force(const MechanicsValues<dim, spacedim> &  m_values,
                  ArrayView<Tensor<1, spacedim, Number>> &forces) // = 0 TODO fix this
    {
      // It shouldn't be possible to get here but since compute_surface_force
      // and compute_volume_force both call this function we need it to make the
      // linker happy
      Assert(false, ExcFDLInternalError());
    }

    /**
     * Compute a surface force. Defaults to calling compute_force.
     */
    virtual void
    compute_surface_force(
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_face_iterator &face,
      ArrayView<Tensor<1, spacedim, Number>> &                           forces)
    {
      compute_force(m_values, forces);
    }

    /**
     * Compute a volume force. Defaults to calling compute_force.
     */
    virtual void
    compute_volume_force(
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      ArrayView<Tensor<1, spacedim, Number>> &                           forces)
    {
      compute_force(m_values, forces);
    }

  protected:
    Quadrature<dim> quadrature;
  };
} // namespace fdl

#endif
