#ifndef included_fiddle_mechanics_force_contribution_lib_h
#define included_fiddle_mechanics_force_contribution_lib_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Base class for spring forces (both boundary and volumetric). This force is
   *
   * F = k (X_ref - X)
   *
   * in which X_ref is the reference position vector stored by this class.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class SpringForceBase : public ForceContribution<dim, spacedim, double>
  {
  public:
    /**
     * Constructor. This class stores a pointer to the DoFHandler so that it can
     * access DoFs on cells and copies the provided references position vector.
     * Applies the force on every cell.
     */
    template <int q_dim>
    SpringForceBase(
      const Quadrature<q_dim>                          &quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
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

    virtual void
    setup_force(
      const double                                      time,
      const LinearAlgebra::distributed::Vector<double> &position,
      const LinearAlgebra::distributed::Vector<double> &velocity) override;

    virtual void
    finish_force(const double time) override;

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

  /**
   * Volumetric spring forces.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class SpringForce : public SpringForceBase<dim, spacedim, Number>
  {
  public:
    /**
     * Constructor. This class stores a pointer to the DoFHandler so that it can
     * access DoFs on cells and copies the provided references position vector.
     * Applies the force on every cell.
     */
    SpringForce(
      const Quadrature<dim>                            &quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    SpringForce(const Quadrature<dim>           &quad,
                const double                     spring_constant,
                const DoFHandler<dim, spacedim> &dof_handler,
                const Mapping<dim, spacedim>    &mapping,
                const Function<spacedim>        &reference_function);

    /**
     * Constructor. Same idea, but only applies the force on cells with the
     * provided material ids.
     *
     * @note if @p material_ids is empty then the force will not be applied on
     * any cell.
     */
    SpringForce(
      const Quadrature<dim>                            &quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const std::vector<types::material_id>            &material_ids,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    SpringForce(const Quadrature<dim>                 &quad,
                const double                           spring_constant,
                const DoFHandler<dim, spacedim>       &dof_handler,
                const Mapping<dim, spacedim>          &mapping,
                const std::vector<types::material_id> &material_ids,
                const Function<spacedim>              &reference_position);

    virtual bool
    is_volume_force() const override;

    virtual void
    compute_volume_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator
        & /*cell*/,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const override;

  protected:
    std::vector<types::material_id> material_ids;
  };

  /**
   * Boundary spring forces.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class BoundarySpringForce : public SpringForceBase<dim, spacedim, Number>
  {
  public:
    /**
     * Constructor. This class stores a pointer to the DoFHandler so that it can
     * access DoFs on cells and copies the provided references position vector.
     * Applies the force on every cell.
     */
    BoundarySpringForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    BoundarySpringForce(const Quadrature<dim - 1>       &quad,
                        const double                     spring_constant,
                        const DoFHandler<dim, spacedim> &dof_handler,
                        const Mapping<dim, spacedim>    &mapping,
                        const Function<spacedim>        &reference_function);

    /**
     * Constructor. Same idea, but only applies the force on faces with the
     * provided boundary ids.
     *
     * @note if @p boundary_ids is empty then the force will not be applied on
     * any boundary.
     */
    BoundarySpringForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const std::vector<types::boundary_id>            &boundary_ids,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    BoundarySpringForce(const Quadrature<dim - 1>             &quad,
                        const double                           spring_constant,
                        const DoFHandler<dim, spacedim>       &dof_handler,
                        const Mapping<dim, spacedim>          &mapping,
                        const std::vector<types::boundary_id> &boundary_ids,
                        const Function<spacedim> &reference_position);

    virtual bool
    is_boundary_force() const override;

    virtual void
    compute_boundary_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_face_iterator &face,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const override;

  protected:
    std::vector<types::boundary_id> boundary_ids;
  };

  /**
   * Velocity damping force: applies a drag force directly proportional to the
   * present velocity field to enforce zero movement:
   *
   * F = -k U
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class DampingForce : public ForceContribution<dim, spacedim, double>
  {
  public:
    /**
     * Constructor.
     */
    DampingForce(const Quadrature<dim> &quad, const double damping_constant);

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

    virtual bool
    is_volume_force() const override;

    virtual void
    compute_volume_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator
        & /*cell*/,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const override;


  protected:
    double damping_constant;
  };

  /**
   * Pericardium Model Boundary Force
   * F = n * ( k_spring * ( X_ref - x ) - k_damping * U ) * n
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class OrthogonalSpringDashpotForce
    : public SpringForceBase<dim, spacedim, Number>
  {
  public:
    /**
     * Constructor. This class stores a pointer to the DoFHandler so that it can
     * access DoFs on cells and copies the provided references position vector.
     * Applies the force on every cell.
     */
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const double                                      damping_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    OrthogonalSpringDashpotForce(const Quadrature<dim - 1> &quad,
                                 const double               spring_constant,
                                 const double               damping_constant,
                                 const DoFHandler<dim, spacedim> &dof_handler,
                                 const Mapping<dim, spacedim>    &mapping,
                                 const Function<spacedim> &reference_function);

    /**
     * Constructor. Same idea, but only applies the force on faces with the
     * provided boundary ids.
     *
     * @note if @p boundary_ids is empty then the force will not be applied on
     * any boundary.
     */
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const double                                      damping_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const std::vector<types::boundary_id>            &boundary_ids,
      const LinearAlgebra::distributed::Vector<double> &reference_position);

    /**
     * Same, but for an initial position set up by a Function instead of a
     * specified vector.
     */
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>             &quad,
      const double                           spring_constant,
      const double                           damping_constant,
      const DoFHandler<dim, spacedim>       &dof_handler,
      const Mapping<dim, spacedim>          &mapping,
      const std::vector<types::boundary_id> &boundary_ids,
      const Function<spacedim>              &reference_position);

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

    /**
     * Define this force as a boundary force
     */
    virtual bool
    is_boundary_force() const override;

    virtual void
    compute_boundary_force(
      const double                          time,
      const MechanicsValues<dim, spacedim> &m_values,
      const typename Triangulation<dim, spacedim>::active_face_iterator &face,
      ArrayView<Tensor<1, spacedim, Number>> &forces) const override;

  protected:
    double damping_constant;

    std::vector<types::boundary_id> boundary_ids;
  };

  /**
   * Modified Neo-Hookean material model.
   *
   * By 'modified', we mean that the first invariant is the modified one -
   * i.e., we use $I1_bar = J^{-2/3} I1$.
   *
   * This leads to a PK1 stress
   *
   *     PP = G J^{-2/3} (FF - I1 / 3 FF^-T)
   *
   * in which G is the shear modulus.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class ModifiedNeoHookeanStress
    : public ForceContribution<dim, spacedim, Number>
  {
  public:
    /**
     * Constructor.
     */
    ModifiedNeoHookeanStress(const Quadrature<dim> &quad,
                             const double           shear_modulus);

    /**
     * Constructor.
     *
     * @note Like elsewhere, if material_ids is empty then this stress will
     * not be used on any cell.
     */
    ModifiedNeoHookeanStress(
      const Quadrature<dim>                 &quad,
      const double                           shear_modulus,
      const std::vector<types::material_id> &material_ids);

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

    /**
     * Define this force as a PK1 stress.
     */
    virtual bool
    is_stress() const override;

    virtual void
    compute_stress(
      const double                          time,
      const MechanicsValues<dim, spacedim> &me_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      ArrayView<Tensor<2, spacedim, Number>> &stresses) const override;

  protected:
    double shear_modulus;

    std::vector<types::material_id> material_ids;
  };

  /**
   * Modified Mooney-Rivlin material model.
   *
   * By 'modified', we mean that the first and second invariants are the
   * modified ones - i.e., we use $I1_bar = J^{-2/3} I1$ and $I2_bar = J^{-4/3}
   * I2$.
   *
   * This leads to a PK1 stress
   *
   *     PP = 2 c_1 J^{-2/3} (FF - I1 / 3 FF^-T) + 2 c_2 J^{-4/3} (I1 FF - FF CC
   * - 2 I2 / 3 FF^-T)
   *
   * in which c_1 and c_2 are material constants.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class ModifiedMooneyRivlinStress
    : public ForceContribution<dim, spacedim, Number>
  {
  public:
    /**
     * Constructor.
     */
    ModifiedMooneyRivlinStress(
      const Quadrature<dim>                 &quad,
      const double                           material_constant_1,
      const double                           material_constant_2);

    /**
     * Constructor.
     *
     * @note Like elsewhere, if material_ids is empty then this stress will
     * not be used on any cell.
     */
    ModifiedMooneyRivlinStress(
      const Quadrature<dim>                 &quad,
      const double                           material_constant_1,
      const double                           material_constant_2,
      const std::vector<types::material_id> &material_ids);

    /**
     * Get the update flags this force contribution requires for MechanicsValues
     * objects.
     */
    virtual MechanicsUpdateFlags
    get_mechanics_update_flags() const override;

    /**
     * Define this force as a PK1 stress.
     */
    virtual bool
    is_stress() const override;

    virtual void
    compute_stress(
      const double                          time,
      const MechanicsValues<dim, spacedim> &me_values,
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      ArrayView<Tensor<2, spacedim, Number>> &stresses) const override;

  protected:
    double material_constant_1;

    double material_constant_2;

    std::vector<types::material_id> material_ids;
  };
} // namespace fdl

#endif
