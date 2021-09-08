#ifndef included_fiddle_mechanics_force_contribution_h
#define included_fiddle_mechanics_force_contribution_h

#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/lac/la_parallel_vector.h>

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
      : is_volumetric(true)
      , cell_quadrature(quad)
    {}

    /**
     * Constructor
     */
    ForceContribution(const Quadrature<dim - 1> &quad)
      : is_volumetric(false)
      , face_quadrature(quad)
    {}

    const Quadrature<dim> &
    get_cell_quadrature() const
    {
      Assert(is_volumetric,
             ExcMessage("This function can only be called on forces "
                        "constructed with a volumetric (codimension zero) "
                        "quadrature rule."));
      return cell_quadrature;
    }

    const Quadrature<dim - 1> &
    get_face_quadrature() const
    {
      Assert(!is_volumetric,
             ExcMessage("This function can only be called on forces "
                        "constructed with a face (codimension one) quadrature"
                        "rule."));
      return face_quadrature;
    }

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const = 0;

    /**
     * Get the update flags this force contribution requires for FEValues
     * objects.
     */
    virtual UpdateFlags
    get_update_flags() const = 0;

    virtual bool
    is_stress() const
    {
      return false;
    }

    virtual bool
    is_boundary_force() const
    {
      return false;
    }

    virtual bool
    is_volume_force() const
    {
      return false;
    }

    /**
     * Some forces that are not defined in a straightforward way (e.g., pressure
     * fields) require additional setup before their force contribution is
     * available at a given time. For example, computing a static pressure
     * requires solving a linear system.
     *
     * To aid in the user definition of such things,
     * IFEDMethod::computeLagrangianForce() and related functions will call this
     * function for each force contribution before calling any of
     * compute_force(), compute_surface_force(), etc. at a specific time.
     */
    virtual void
    setup_force(const double                                      time,
                const LinearAlgebra::distributed::Vector<double> &position,
                const LinearAlgebra::distributed::Vector<double> &velocity)
    {
      (void)time;
      (void)position;
      (void)velocity;
    }

    /**
     * Matching function to setup_force() which is instead called after all
     * required forces have been computed. This function may be used to
     * deallocate memory or perform any other necessary cleanup.
     */
    virtual void
    finish_force(const double time)
    {
      (void)time;
    }

    /**
     * Compute forces at quadrature points. Should work regardless of whether
     * we are on the surface of the element or inside it.
     */
    virtual void
    compute_force(
      const double                            time,
      const MechanicsValues<dim, spacedim> &  m_values,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const // = 0 TODO fix this
    {
      // It shouldn't be possible to get here but since compute_surface_force
      // and compute_volume_force both call this function we need it to make the
      // linker happy
      (void)time;
      (void)m_values;
      (void)forces;
      Assert(false, ExcFDLInternalError());
    }

    /**
     * Compute a surface force. Defaults to calling compute_force.
     */
    virtual void
    compute_surface_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_face_iterator
        & /*face*/,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const
    {
      compute_force(time, m_values, forces);
    }

    /**
     * Compute a volume force. Defaults to calling compute_force.
     */
    virtual void
    compute_volume_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator
        & /*cell*/,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const
    {
      compute_force(time, m_values, forces);
    }

    virtual void
    compute_stress(const double                            time,
                   const MechanicsValues<dim, spacedim> &  me_values,
                   ArrayView<Tensor<2, spacedim, Number>> &stresses) const
    {
      (void)time;
      (void)me_values;
      (void)stresses;
      Assert(false, ExcFDLInternalError());
    }

  private:
    bool is_volumetric;

    Quadrature<dim> cell_quadrature;

    Quadrature<dim - 1> face_quadrature;
  };
} // namespace fdl

#endif
