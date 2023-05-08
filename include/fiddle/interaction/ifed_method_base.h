#ifndef included_fiddle_interaction_ifed_method_base_h
#define included_fiddle_interaction_ifed_method_base_h

#include <fiddle/base/config.h>

#include <fiddle/mechanics/part.h>
#include <fiddle/mechanics/part_vectors.h>

#include <ibamr/IBStrategy.h>

#include <ibtk/SAMRAIDataCache.h>

#include <BasePatchHierarchy.h>
#include <tbox/Pointer.h>

#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   */
  template <int dim, int spacedim = dim>
  class IFEDMethodBase : public IBAMR::IBStrategy
  {
  public:
    static_assert(spacedim == NDIM, "Only available for spacedim == NDIM");

    IFEDMethodBase(const std::string                     &object_name,
                   std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
                   std::vector<Part<dim, spacedim>>     &&input_parts,
                   const bool register_for_restart = true);

    /**
     * Destructor.
     */
    ~IFEDMethodBase();

    /**
     * @name fluid-structure interaction.
     * @{
     */
    virtual double
    getMaxPointDisplacement() const override;

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
            &u_ghost_fill_scheds,
      int    integrator_step,
      double init_data_time,
      bool   initial_time) override;
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
     * @name timestepping.
     * @{
     */
    virtual void
    preprocessIntegrateData(double current_time,
                            double new_time,
                            int    num_cycles) override;

    virtual void
    postprocessIntegrateData(double current_time,
                             double new_time,
                             int    num_cycles) override;

    virtual void
    forwardEulerStep(double current_time, double new_time) override;

    virtual void
    backwardEulerStep(double current_time, double new_time) override;

    virtual void
    midpointStep(double current_time, double new_time) override;

    virtual void
    trapezoidalStep(double current_time, double new_time) override;
    /**
     * @}
     */

    /**
     * @}
     */

    /**
     * @name book-keeping.
     * @{
     */
    virtual void
    putToDatabase(tbox::Pointer<tbox::Database> db) override;
    /**
     * @}
     */

  protected:
    /**
     * Book-keeping
     * @{
     */
    std::string object_name;

    bool register_for_restart;

    bool started_time_integration;

    double current_time;
    double half_time;
    double new_time;
    /**
     * @}
     */

    /**
     * Finite difference data structures
     * @{
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy;

    std::shared_ptr<IBTK::SAMRAIDataCache> eulerian_data_cache;
    /**
     * @}
     */

    /**
     * Finite element data structures
     * @{
     */
    std::vector<Part<dim, spacedim>> parts;

    std::vector<Part<dim - 1, spacedim>> surface_parts;

    PartVectors<dim, spacedim> part_vectors;

    PartVectors<dim - 1, spacedim> surface_part_vectors;

    std::deque<LinearAlgebra::distributed::Vector<double>>
      positions_at_last_regrid;

    std::deque<LinearAlgebra::distributed::Vector<double>>
      surface_positions_at_last_regrid;
    /**
     * @}
     */
  };
} // namespace fdl

#endif
