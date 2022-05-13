#ifndef included_fiddle_interaction_dlm_method_h
#define included_fiddle_interaction_dlm_method_h

#include <fiddle/mechanics/force_contribution_lib.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/la_parallel_vector.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Abstract class implementing an interface for the distributed Lagrange
   * multiplier method.
   */
  template <int dim, int spacedim = dim>
  class DLMMethodBase : public Subscriptor
  {
  public:
    /**
     * Get the position of the structure (from the point of view of this object)
     * at the specified time. This may involve interpolation in time.
     */
    virtual void
    get_mechanics_position(
      const double                                time,
      LinearAlgebra::distributed::Vector<double> &position) const = 0;

    /**
     * Get a reference to the current position, whereever it may be. Useful for
     * initializing other classes that need some position vector to make sense.
     */
    virtual const LinearAlgebra::distributed::Vector<double> &
    get_current_mechanics_position() const = 0;

    /**
     * Get the velocity of the structure (from the point of view of this object)
     * at the specified time. This may involve interpolation in time.
     */
    virtual void
    get_mechanics_velocity(
      const double                                time,
      LinearAlgebra::distributed::Vector<double> &velocity) const = 0;

    /**
     * Get a reference to the current velocity, whereever it may be. Useful for
     * initializing other classes that need some velocity vector to make sense.
     */
    virtual const LinearAlgebra::distributed::Vector<double> &
    get_current_mechanics_velocity() const = 0;
  };

  /**
   * Force contribution based on a DLMMethod.
   */
  template <int dim, int spacedim = dim>
  class DLMForce : public ForceContribution<dim>
  {
  public:
    DLMForce(const Quadrature<dim> &          quad,
             const double                     spring_constant,
             const double                     damping_constant,
             const DoFHandler<dim, spacedim> &dof_handler,
             const DLMMethodBase<dim, spacedim> &   dlm);

    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

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
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      ArrayView<Tensor<1, spacedim>>       &forces) const override;

  protected:
    double spring_constant;
    double damping_constant;

    SmartPointer<const DoFHandler<dim, spacedim>> dof_handler;
    SmartPointer<const DLMMethodBase<dim, spacedim>> dlm;

    SmartPointer<const LinearAlgebra::distributed::Vector<double>> current_position;
    SmartPointer<const LinearAlgebra::distributed::Vector<double>> current_velocity;

    LinearAlgebra::distributed::Vector<double> reference_position;
    LinearAlgebra::distributed::Vector<double> reference_velocity;

    mutable std::vector<types::global_dof_index> scratch_cell_dofs;
    mutable std::vector<double>                  scratch_dof_values;
    mutable std::vector<Tensor<1, dim>>          scratch_qp_values;
  };
} // namespace fdl

#endif
