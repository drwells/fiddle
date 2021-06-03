#ifndef included_fiddle_interaction_ifed_method_h
#define included_fiddle_interaction_ifed_method_h

#include <fiddle/base/exceptions.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/mechanics/part.h>

#include <ibamr/IBStrategy.h>

#include <ibtk/LEInteractor.h>
#include <ibtk/SecondaryHierarchy.h>

#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Class implementing the volumetric IFED method.
   *
   * @note This class only makes sense when spacedim (the template parameter) is
   * equal to NDIM (the IBAMR spatial dimension macro). Like elsewhere in the
   * library, since template parameters are preferrable to macros, while the two
   * are equal we use spacedim whenever possible.
   */
  template <int dim, int spacedim = dim>
  class IFEDMethod : public IBAMR::IBStrategy
  {
  public:
    static_assert(spacedim == NDIM, "Only available for spacedim == NDIM");

    /**
     * @name Constructors.
     * @{
     */

    /**
     * Constructor. Assumes ownership of the provided parts.
     */
    IFEDMethod(tbox::Pointer<tbox::Database>      input_db,
               std::vector<Part<dim, spacedim>> &&input_parts);

    /**
     * @}
     */

    /**
     * @name Initialization.
     * @{
     */

    /**
     * Initialize Lagrangian data corresponding to the given AMR patch hierarchy
     * at the start of a computation. This may involve reading restart data.
     */
    virtual void
    initializePatchHierarchy(
      tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
      tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg,
      int                                              u_data_index,
      const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
        &u_synch_scheds,
      const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
        &    u_ghost_fill_scheds,
      int    integrator_step,
      double init_data_time,
      bool   initial_time) override;
    /**
     * @}
     */

    /**
     * @name fluid-structure interaction.
     * @{
     */
    virtual void
    interpolateVelocity(
      int u_data_index,
      const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
        &u_synch_scheds,
      const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
        &    u_ghost_fill_scheds,
      double data_time) override;

    virtual void
    spreadForce(int                               f_data_index,
                IBTK::RobinPhysBdryPatchStrategy *f_phys_bdry_op,
                const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
                  &    f_prolongation_scheds,
                double data_time) override;

    /**
     * Tag cells in @p hierarchy that intersect with the structure.
     */
    virtual void
    applyGradientDetector(
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
      int                                               level_number,
      double                                            error_data_time,
      int                                               tag_index,
      bool                                              initial_time,
      bool uses_richardson_extrapolation_too) override;

    /**
     * @}
     */

    /**
     * @name timestepping.
     * @{
     */
    virtual void
    preprocessIntegrateData(double current_time,
                            double new_time,
                            int /*num_cycles*/) override
    {
      this->current_time = current_time;
      this->new_time     = new_time;
      this->half_time    = current_time + 0.5 * (new_time - current_time);
    }

    virtual void
    postprocessIntegrateData(double /*current_time*/,
                             double /*new_time*/,
                             int /*num_cycles*/) override
    {
      this->current_time = std::numeric_limits<double>::quiet_NaN();
      this->new_time     = std::numeric_limits<double>::quiet_NaN();
      this->half_time    = std::numeric_limits<double>::quiet_NaN();

      half_position_vectors.clear();
      new_position_vectors.clear();
      half_velocity_vectors.clear();
      new_velocity_vectors.clear();
    }

    virtual void
    forwardEulerStep(double current_time, double new_time) override
    {
      (void)current_time;
      (void)new_time;
    }

    virtual void
    backwardEulerStep(double current_time, double new_time) override
    {
      (void)current_time;
      (void)new_time;
    }

    virtual void
    midpointStep(double current_time, double new_time) override
    {
      (void)current_time;
      (void)new_time;
    }

    virtual void
    trapezoidalStep(double current_time, double new_time) override
    {
      (void)current_time;
      (void)new_time;
    }

    virtual void
    computeLagrangianForce(double data_time) override
    {
      (void)data_time;
    }
    /**
     * @}
     */

    /**
     * @name Parallel data distribution.
     * @{
     */
    virtual void
    beginDataRedistribution(
      tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
      tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg) override;

    virtual void
    endDataRedistribution(
      tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
      tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg) override;
    /**
     * @}
     */

    /**
     * @name book-keeping.
     * @{
     */
    virtual const hier::IntVector<spacedim> &
    getMinimumGhostCellWidth() const override;

    void
    registerEulerianVariables() override;
    /**
     * @}
     */

  protected:
    /**
     * Actually set up the interaction objects - will ultimately support nodal
     * and elemental, but other interaction objects can be implemented here by
     * overriding this function.
     */
    virtual void
    reinit_interactions();

    /**
     * Book-keeping
     * @{
     */
    tbox::Pointer<tbox::Database> input_db;

    // Get the correct temporary vector or (if time == current_time) the
    // vector from the Part object
    const LinearAlgebra::distributed::Vector<double> &
    get_position(const unsigned int part_n, const double time)
    {
      if (std::abs(time - current_time) < 1e-12)
        return parts[part_n].get_position();
      if (std::abs(time - half_time) < 1e-12)
        {
          Assert(part_n < half_position_vectors.size(),
                 ExcMessage(
                   "The requested position vector has not been calculated."));
          return half_position_vectors[part_n];
        }
      if (std::abs(time - new_time) < 1e-12)
        {
          Assert(part_n < new_position_vectors.size(),
                 ExcMessage(
                   "The requested position vector has not been calculated."));
          return new_position_vectors[part_n];
        }

      Assert(false, ExcFDLInternalError());
      return parts[part_n].get_position();
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity(const unsigned int part_n, const double time)
    {
      if (std::abs(time - current_time) < 1e-12)
        return parts[part_n].get_velocity();
      if (std::abs(time - half_time) < 1e-12)
        {
          Assert(part_n < half_velocity_vectors.size(),
                 ExcMessage(
                   "The requested velocity vector has not been calculated."));
          return half_velocity_vectors[part_n];
        }
      if (std::abs(time - new_time) < 1e-12)
        {
          Assert(part_n < new_velocity_vectors.size(),
                 ExcMessage(
                   "The requested velocity vector has not been calculated."));
          return new_velocity_vectors[part_n];
        }

      Assert(false, ExcFDLInternalError());
      return parts[part_n].get_position();
    }

    double current_time;
    double half_time;
    double new_time;
    /**
     * @}
     */

    /**
     * Finite element data structures
     * @{
     */
    std::vector<Part<dim, spacedim>> parts;

    std::vector<LinearAlgebra::distributed::Vector<double>>
      half_position_vectors;
    std::vector<LinearAlgebra::distributed::Vector<double>>
      new_position_vectors;

    std::vector<LinearAlgebra::distributed::Vector<double>>
      half_velocity_vectors;
    std::vector<LinearAlgebra::distributed::Vector<double>>
      new_velocity_vectors;

    std::vector<LinearAlgebra::distributed::Vector<double>> half_force_vectors;
    std::vector<LinearAlgebra::distributed::Vector<double>> new_force_vectors;
    /**
     * @}
     */

    /**
     * Finite difference data structures
     * @{
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> primary_hierarchy;

    IBTK::SecondaryHierarchy secondary_hierarchy;

    int lagrangian_workload_current_index = IBTK::invalid_index;
    int lagrangian_workload_new_index     = IBTK::invalid_index;
    int lagrangian_workload_scratch_index = IBTK::invalid_index;

    tbox::Pointer<hier::Variable<spacedim>> lagrangian_workload_var;
    /**
     * @}
     */

    /**
     * Interaction data structures
     * @{
     */
    std::vector<std::unique_ptr<InteractionBase<dim, spacedim>>> interactions;
    /**
     * @}
     */
  };
} // namespace fdl

#endif
