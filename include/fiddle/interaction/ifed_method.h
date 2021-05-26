#ifndef included_fiddle_interaction_ifed_method_h
#define included_fiddle_interaction_ifed_method_h

#include <fiddle/base/exceptions.h>

#include <ibamr/IBStrategy.h>

#include <ibtk/LEInteractor.h>

#include <fiddle/mechanics/part.h>

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
    IFEDMethod(std::vector<Part<dim, spacedim>> &&input_parts)
      : parts(std::move(input_parts))
    {}

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
      int                                              u_data_idx,
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
      int u_data_idx,
      const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
        &u_synch_scheds,
      const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
        &    u_ghost_fill_scheds,
      double data_time) override;

    virtual void
    spreadForce(int                               f_data_idx,
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
      tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg) override
    {
      (void)hierarchy;
      (void)gridding_alg;
    }

    virtual void
    endDataRedistribution(
      tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
      tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg) override
    {
      (void)hierarchy;
      (void)gridding_alg;
    }
    /**
     * @}
     */

    /**
     * @name book-keeping.
     * @{
     */
    virtual const hier::IntVector<spacedim> &
    getMinimumGhostCellWidth() const override
    {
      // Like elsewhere, we are hard-coding in bspline 3 for now
      const std::string kernel_name = "BSPLINE_3";
      const int         ghost_width =
        IBTK::LEInteractor::getMinimumGhostWidth(kernel_name);
      static hier::IntVector<spacedim> gcw;
      for (int i = 0; i < spacedim; ++i)
        gcw[i] = ghost_width;
      return gcw;
    }
    /**
     * @}
     */

  protected:
    std::vector<Part<dim, spacedim>> parts;
  };
} // namespace fdl

#endif
