#ifndef included_fiddle_interaction_ifed_method_h
#define included_fiddle_interaction_ifed_method_h

#include <fiddle/base/exceptions.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/mechanics/part.h>
#include <fiddle/mechanics/part_vectors.h>

#include <ibamr/IBStrategy.h>

#include <ibtk/LEInteractor.h>
#include <ibtk/SAMRAIDataCache.h>
#include <ibtk/SAMRAIGhostDataAccumulator.h>
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
   *
   * <h2>Options read from the input database</h2>
   * <ul>
   *   <li>enable_logging: whether or not to log things like the workload.
   *     Defaults to FALSE.</li>
   *   <li>log_solver_iterations: whether or not to log number of iterations
   *     required for finite element solvers. Defaults to FALSE.</li>
   *   <li>skip_initial_workload: whether to skip printing the initial workload,
   *     to work around an issue with SAMRAI. This is typically not necessary to
   *     set inside user codes. Defaults to FALSE.</li>
   *   <li>GriddingAlgorithm: Database for setting up the internal
   *     GriddingAlgorithm object.</li>
   *   <li>LoadBalancer: Database for setting up the internal LoadBalancer
   *     object.</li>
   * </ul>
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
    IFEDMethod(const std::string &                object_name,
               tbox::Pointer<tbox::Database>      input_db,
               std::vector<Part<dim, spacedim>> &&input_parts,
               const bool                         register_for_restart = true);

    /**
     * Destructor.
     */
    ~IFEDMethod();

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
                            int /*num_cycles*/) override;

    virtual void
    postprocessIntegrateData(double /*current_time*/,
                             double /*new_time*/,
                             int /*num_cycles*/) override;

    virtual void
    forwardEulerStep(double current_time, double new_time) override;

    virtual void
    backwardEulerStep(double current_time, double new_time) override;

    virtual void
    midpointStep(double current_time, double new_time) override;

    virtual void
    trapezoidalStep(double current_time, double new_time) override;

    virtual void
    computeLagrangianForce(double data_time) override;
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
    virtual void
    putToDatabase(tbox::Pointer<tbox::Database> db) override;

    virtual const hier::IntVector<spacedim> &
    getMinimumGhostCellWidth() const override;

    void
    registerEulerianVariables() override;
    /**
     * @}
     */

    const Part<dim, spacedim> &
    get_part(const unsigned int part_n) const;

    int
    get_lagrangian_workload_current_index() const
    {
      Assert(lagrangian_workload_plot_index != IBTK::invalid_index,
             ExcMessage("The Lagrangian workload index has not yet been set."));
      return lagrangian_workload_plot_index;
    }

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
    std::string object_name;

    bool register_for_restart;

    tbox::Pointer<tbox::Database> input_db;

    std::vector<std::string> ib_kernels;

    bool started_time_integration;

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

    PartVectors<dim, spacedim> part_vectors;
    /**
     * @}
     */

    /**
     * Finite difference data structures
     * @{
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> primary_hierarchy;

    std::shared_ptr<IBTK::SAMRAIDataCache> primary_eulerian_data_cache;

    IBTK::SecondaryHierarchy secondary_hierarchy;

    int lagrangian_workload_plot_index = IBTK::invalid_index;

    int lagrangian_workload_current_index = IBTK::invalid_index;
    int lagrangian_workload_new_index     = IBTK::invalid_index;
    int lagrangian_workload_scratch_index = IBTK::invalid_index;

    tbox::Pointer<hier::Variable<spacedim>> lagrangian_workload_var;

    std::unique_ptr<IBTK::SAMRAIGhostDataAccumulator> ghost_data_accumulator;
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

  // Inline functions
  template <int dim, int spacedim>
  inline const Part<dim, spacedim> &
  IFEDMethod<dim, spacedim>::get_part(const unsigned int part_n) const
  {
    AssertIndexRange(part_n, parts.size());
    return parts[part_n];
  }

} // namespace fdl

#endif
