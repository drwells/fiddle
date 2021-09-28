#ifndef included_fiddle_mechanics_force_contribution_lib_h
#define included_fiddle_mechanics_force_contribution_lib_h

#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

namespace fdl
{
  using namespace dealii;

  /**
   * A class representing a spring force from a specified finite element
   * position vector. The force is
   *
   * F = k (X_ref - X)
   *
   * where X_ref is the vector stored by this class.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class SpringForce : public ForceContribution<dim, spacedim, double>
  {
  public:
    /**
     * Constructor. This class stores a pointer to the DoFHandler so that it can
     * access DoFs on cells and copies the provided references position vector.
     */
    SpringForce(
      const Quadrature<dim> &                           quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim> &                 dof_handler,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Set the reference position to a new value.
     */
    void
    set_reference_position(
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

    /**
     * Get the update flags this force contribution requires for FEValues
     * objects.
     */
    virtual UpdateFlags
    get_update_flags() const override;

    virtual bool
    is_volume_force() const override;

    virtual void
    setup_force(
      const double                                      time,
      const LinearAlgebra::distributed::Vector<double> &position,
      const LinearAlgebra::distributed::Vector<double> &velocity) override;

    virtual void
    finish_force(const double time) override;

    virtual void
    compute_volume_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator
        & /*cell*/,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const override;

  protected:
    double spring_constant;

    SmartPointer<const DoFHandler<dim, spacedim>> dof_handler;

    SmartPointer<const LinearAlgebra::distributed::Vector<double>>
                                               current_position;
    LinearAlgebra::distributed::Vector<double> reference_position;

    mutable std::vector<types::global_dof_index> scratch_cell_dofs;
    mutable std::vector<double>                  scratch_dof_values;
    mutable std::vector<Tensor<1, spacedim>>     scratch_qp_values;
  };
} // namespace fdl

#endif
