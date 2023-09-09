#ifndef included_fiddle_interaction_interaction_base_h
#define included_fiddle_interaction_interaction_base_h

#include <fiddle/base/config.h>

#include <fiddle/grid/overlap_tria.h>

#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <tbox/Pointer.h>

#include <memory>
#include <utility>
#include <vector>

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class PatchHierarchy;
  }
} // namespace SAMRAI

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Many interaction operations require multiple computation and
   * communication steps - since it might be useful, in an application, to
   * interleave these, these steps are broken up into distinct member function
   * calls of the Interaction class. However, since each function call leaves
   * the computation in an intermediate step, this class' job is to
   * encapsulate that state.
   *
   *
   */
  struct TransactionBase
  {
    virtual ~TransactionBase() = default;

    /**
     * Delegate responsibility for completing all outstanding MPI requests to
     * some other object (i.e., someone else will call MPI_Waitall() or an
     * equivalent function). The corresponding requests owned by this object
     * will be set to MPI_REQUEST_NULL (i.e., completed requests).
     *
     * When doing multiple concurrent global to overlap scatters, the sum of the
     * scatters is load balanced but individual scatters are not. Hence it is
     * more efficient to wait for all scatters simultaneously than individually.
     */
    virtual std::vector<MPI_Request>
    delegate_outstanding_requests();
  };

  /**
   * Standard class for transactions - used by InteractionBase,
   * ElementalInteraction and NodalInteraction.
   *
   * @note Several of the arrays owned by this class will be asynchronously
   * written into by MPI - moving or resizing these arrays can result in program
   * crashes. It should normally not be necessary for objects that do not set up
   * a transaction to modify it.
   */
  template <int dim, int spacedim = dim>
  struct Transaction : public TransactionBase
  {
    /// Name of the IB kernel we should use.
    std::string kernel_name;

    /// Current patch index.
    int current_data_idx;

    /// Native position DoFHandler.
    SmartPointer<const DoFHandler<dim, spacedim>> native_position_dof_handler;

    /// position scatter.
    Scatter<double> position_scatter;

    /// Native-partitioned position.
    SmartPointer<const LinearAlgebra::distributed::Vector<double>>
      native_position;

    /// Overlap-partitioned position.
    Vector<double> overlap_position;

    /// Native DoFHandler.
    SmartPointer<const DoFHandler<dim, spacedim>> native_dof_handler;

    /// The other scatter (used for spreading).
    Scatter<double> solution_scatter;

    /// The other scatter (used for assembly).
    Scatter<double> rhs_scatter;

    /// The operation used in the scatter.
    VectorOperation::values rhs_scatter_back_op;

    /// Mapping to use for the provided finite element field.
    SmartPointer<const Mapping<dim, spacedim>> mapping;

    /// Native-partitioned vector used for assembly.
    SmartPointer<LinearAlgebra::distributed::Vector<double>> native_rhs;

    /// Native-partitioned vector used for spreading.
    SmartPointer<const LinearAlgebra::distributed::Vector<double>>
      native_solution;

    /// Overlap-partitioned vector used for assembly.
    Vector<double> overlap_rhs;

    /// Overlap-partitioned vector used for spreading.
    Vector<double> overlap_solution;

    /// Possible states for a transaction.
    enum class State
    {
      ScatterStart,
      ScatterFinish,
      Intermediate,
      AccumulateStart,
      AccumulateFinish,
      Done
    };

    /// Next state. Used for consistency checking.
    State next_state;

    /// Possible operations.
    enum class Operation
    {
      Interpolation,
      Spreading
    };

    /// Operation of the current transaction. Used for consistency checking.
    Operation operation;

    virtual std::vector<MPI_Request>
    delegate_outstanding_requests() override;
  };

  /**
   * Transaction class used for workload calculations.
   *
   * @note Several of the arrays owned by this class will be asynchronously
   * written into by MPI - moving or resizing these arrays can result in program
   * crashes. It should normally not be necessary for objects that do not set up
   * a transaction to modify it.
   */
  template <int dim, int spacedim>
  struct WorkloadTransaction : public TransactionBase
  {
    int workload_index;

    /// Native position DoFHandler.
    SmartPointer<const DoFHandler<dim, spacedim>> native_position_dof_handler;

    /// position scatter.
    Scatter<double> position_scatter;

    /// Native-partitioned position.
    SmartPointer<const LinearAlgebra::distributed::Vector<double>>
      native_position;

    /// Overlap-partitioned position.
    Vector<double> overlap_position;

    /// Possible states for a transaction.
    enum class State
    {
      ScatterStart,
      ScatterFinish,
      Intermediate,
      AccumulateStart,
      AccumulateFinish,
      Done
    };

    /// Next state. Used for consistency checking.
    State next_state;

    virtual std::vector<MPI_Request>
    delegate_outstanding_requests() override;
  };

  /**
   * Base class managing interaction between SAMRAI and deal.II data structures,
   * by interpolation and spreading, where the position of the structure is
   * described by a finite element field. This class sets up the data structures
   * and communication patterns necessary for all types of interaction (like
   * nodal or elemental coupling).
   *
   * SAMRAI and deal.II use independent parallel load balancing strategies. In
   * practice this means that operations which require access to both sets of
   * data can be load balanced arbitrarily badly (e.g., SAMRAI may generate a
   * single Patch covering all of the FE data, which would require a single
   * processor to do 100% of the interaction work). This problem is mostly
   * mitigated by the use of OverlapTriangulation here and
   * IBTK::SecondaryHierarchy in IFEDMethod, which together load balance the
   * coupling (interpolation and spreading) operations. This secondary data
   * partitioning results in some additional complexity in moving data between
   * different representations. To achieve good load balancing, communication is
   * split into two steps for moving between the 'native' and 'overlap'
   * partitioning in each direction, which results in five steps for
   * interpolation and four for spreading. This complexity is handled by
   * IFEDMethod: for the most part, inheriting classes should only need to
   * modify the 'intermediate' functions which do the actual computations.
   */
  template <int dim, int spacedim = dim>
  class InteractionBase
  {
  public:
    /**
     * Default constructor. Sets up an empty object.
     */
    InteractionBase();

    /**
     * Constructor. This call is collective.
     *
     * @param[in] input_db Input database. The only value read from the database
     *            is ghost_cell_fraction, which controls the fraction of ghost
     *            cells added to each patch boundary box for the purposes of
     *            associating nodes or elements with a given patch. The default
     *            value is 1.0, which is typically the correct value for
     *            problems with moving meshes.
     *
     * @param[in] native_tria The Triangulation used to define the finite
     *            element fields. This class will use the same MPI communicator
     *            as the one used by this Triangulation.
     *
     * @param[in] active_cell_bboxes Bounding box for each active cell (not just
     *            cells owned by the current processor). This should be computed
     *            with the finite element description of the displacement.
     *
     * @param[in] active_cell_lengths Length scale for each element - usually
     *            used to determine which quadrature rule should be used. Also
     *            for each active cell. This should be computed with the finite
     *            element description of the displacement.
     *
     * @param[inout] patch_hierarchy The patch hierarchy with which we will
     *               interact (i.e., for spreading and interpolation).
     *
     * @param[in] level_numbers Level numbers (coarsest and finest) on which we
     *            are interacting. Presently nodal interaction supports multiple
     *            levels.
     *
     * @param[inout] eulerian_data_cache Pointer to the shared cache of
     *               scratch patch indices of @p patch_hierarchy.
     */
    InteractionBase(
      const tbox::Pointer<tbox::Database>                  &input_db,
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
      const std::vector<float>                             &active_cell_lengths,
      tbox::Pointer<hier::PatchHierarchy<spacedim>>         patch_hierarchy,
      const std::pair<int, int>                            &level_numbers);

    /**
     * Reinitialize the object. Same as the constructor.
     */
    virtual void
    reinit(const tbox::Pointer<tbox::Database>                  &input_db,
           const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>> &active_cell_bboxes,
           const std::vector<float>                        &active_cell_lengths,
           tbox::Pointer<hier::PatchHierarchy<spacedim>>    patch_hierarchy,
           const std::pair<int, int>                       &level_number);

    /**
     * Destructor.
     */
    virtual ~InteractionBase();

    /**
     * Store a pointer to @p native_dof_handler and also compute the
     * equivalent DoFHandler on the overlapping partitioning.
     *
     * This call is collective over the communicator used by this class.
     */
    virtual void
    add_dof_handler(const DoFHandler<dim, spacedim> &native_dof_handler);

    /**
     * For some interactions, the projection operator is actually interpolation.
     * In that case the RHS is the solution and we can skip a lot of work.
     * Defaults to returning false.
     */
    virtual bool
    projection_is_interpolation() const;

    /**
     * Start the computation of the RHS vector corresponding to projecting @p
     * data_idx onto the finite element space specified by @p dof_handler. Since
     * interpolation requires multiple data transfers it is split into five
     * parts. In particular, this first function begins the asynchronous scatter
     * from the native representation to the overlapping representation (the
     * 'scatter' direction).
     *
     * @return This function returns a Transaction object which completely
     * encapsulates the current state of the interpolation.
     *
     * @warning The Transaction returned by this method stores pointers to all
     * of the input arguments. Those pointers must remain valid until after
     * compute_projection_rhs_accumulate_finish() is called.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_scatter_start(
      const std::string                                &kernel_name,
      const int                                         data_idx,
      const DoFHandler<dim, spacedim>                  &position_dof_handler,
      const LinearAlgebra::distributed::Vector<double> &position,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const Mapping<dim, spacedim>                     &mapping,
      LinearAlgebra::distributed::Vector<double>       &rhs);

    /**
     * Finish the scatter to the overlap representation for computing the RHS.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_scatter_finish(
      std::unique_ptr<TransactionBase> transaction) const;

    /**
     * Middle part of velocity interpolation - finalizes the scatters and then
     * performs the actual computations.
     *
     * @note this function does not compute anything - inheriting classes should
     * reimplement this method to set up the RHS in the desired way.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const;

    /**
     * This step begins accumulation of the RHS vector computed in the overlap
     * representation back to the native representation.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_accumulate_start(
      std::unique_ptr<TransactionBase> transaction) const;

    /**
     * Finish the computation of the RHS vector corresponding to projecting @p
     * data_idx onto the finite element space specified by @p dof_handler.
     */
    virtual void
    compute_projection_rhs_accumulate_finish(
      std::unique_ptr<TransactionBase> transaction);

    /**
     * Start spreading from the provided finite element field @p solution by
     * adding them onto the SAMRAI data index @p data_idx.
     *
     * Since, for multi-part models, many different objects may add forces into
     * @p data_idx, at the end of the three spread functions forces may be
     * spread into ghost regions (both between patches and outside the physical
     * domain). The caller must use, e.g., IBTK::RobinPhysBdryPatchStrategy and
     * IBTK::SAMRAIGhostDataAccumulator (in that order) to communicate spread
     * values onto their owning cells.
     *
     * @warning The Transaction returned by this method stores pointers to all
     * of the input arguments. Those pointers must remain valid until after
     * compute_spread_finish() is called.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_spread_scatter_start(
      const std::string                                &kernel_name,
      const int                                         data_idx,
      const LinearAlgebra::distributed::Vector<double> &position,
      const DoFHandler<dim, spacedim>                  &position_dof_handler,
      const Mapping<dim, spacedim>                     &mapping,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const LinearAlgebra::distributed::Vector<double> &solution);

    /**
     * Finish the scatter to the overlap representation for spreading.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_spread_scatter_finish(
      std::unique_ptr<TransactionBase> transaction) const;

    /**
     * Middle part of spreading - performs the actual computations and does not
     * communicate.
     *
     * @note this routine is not const because it must modify the PatchHierarchy
     * stored by the PatchMap.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_spread_intermediate(
      std::unique_ptr<TransactionBase> spread_transaction);

    /**
     * Finish spreading from the provided finite element field @p F. Since no FE
     * data needs to be accumulated (only FD data) this function does not
     * communicate and, unlike interpolation, can therefore happen in one step.
     */
    virtual void
    compute_spread_finish(std::unique_ptr<TransactionBase> spread_transaction);

    /**
     * Add the workload (typically the number of quadrature points) estimate per
     * Eulerian cell into the data index @p workload_index. Has a default empty
     * implementation.
     *
     * @note This method is not const because it must modify the patch index @p
     * workload_index in the stored hierarchy.
     */
    virtual std::unique_ptr<TransactionBase>
    add_workload_start(
      const int                                         workload_index,
      const LinearAlgebra::distributed::Vector<double> &position,
      const DoFHandler<dim, spacedim>                  &position_dof_handler);

    virtual std::unique_ptr<TransactionBase>
    add_workload_intermediate(std::unique_ptr<TransactionBase> t_ptr);

    virtual void
    add_workload_finish(std::unique_ptr<TransactionBase> t_ptr);

  protected:
    /**
     * One difficulty with the way communication is implemented in deal.II is
     * that there are some hard-coded limits on the number of messages that can
     * be posted at once - for example, we can only use 200 channels in
     * LA::d::Vector. A second difficulty is that since that communication
     * happens inside this object we have no way of picking globally unique
     * channel values.
     *
     * Sidestep this completely by doing all the communication for this object
     * over our own communicator. While creating thousands of communicators is
     * likely to be problematic (long set up times, running out of communicator
     * IDs in some MPI implementations, etc.) we will probably not create more
     * than a few dozen of these objects over the course of a simulator run so
     * its unlikely to be a problem.
     */
    MPI_Comm communicator;

    /**
     * Return a reference to the overlap dof handler corresponding to the
     * provided native dof handler.
     */
    DoFHandler<dim, spacedim> &
    get_overlap_dof_handler(
      const DoFHandler<dim, spacedim> &native_dof_handler);

    /**
     * Return a constant reference to the corresponding overlap dof handler.
     */
    const DoFHandler<dim, spacedim> &
    get_overlap_dof_handler(
      const DoFHandler<dim, spacedim> &native_dof_handler) const;

    /**
     * Get the RHS scatter back operation when setting up the transactions. For
     * nodal 'projection' (equivalent to interpolation) this is the max
     * operation (since we set values) and for elemental projection this is
     * addition.
     *
     * The default implementation returns VectorOperation::unknown.
     */
    virtual VectorOperation::values
    get_rhs_scatter_type() const;

    /**
     * Return a scatter corresponding to the provided native dof handler.
     */
    Scatter<double>
    get_scatter(const DoFHandler<dim, spacedim> &native_dof_handler);

    /**
     * Re-cache a Scatter object.
     */
    void
    return_scatter(const DoFHandler<dim, spacedim> &native_dof_handler,
                   Scatter<double>                &&scatter);

    /**
     * @name Geometric data.
     * @{
     */

    /**
     * Native triangulation, which is stored separately.
     */
    SmartPointer<const parallel::shared::Triangulation<dim, spacedim>>
      native_tria;

    /**
     * Overlap triangulation - i.e., the part of native_tria that intersects the
     * patches in patch_level stored on the current processor.
     */
    OverlapTriangulation<dim, spacedim> overlap_tria;

    /**
     * Pointer to the patch hierarchy.
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy;

    /**
     * Number of the patch level we interact with.
     */
    std::pair<int, int> level_numbers;

    /**
     * @}
     */

    /**
     * @name DoF (Eulerian and Lagrangian) data.
     * @{
     */

    /**
     * Pointers to DoFHandlers using native_tria which have equivalent overlap
     * DoFHandlers.
     */
    std::vector<SmartPointer<const DoFHandler<dim, spacedim>>>
      native_dof_handlers;

    /**
     * DoFHandlers defined on the overlap tria, which are equivalent to those
     * stored by @p native_dof_handlers.
     */
    std::vector<std::unique_ptr<DoFHandler<dim, spacedim>>>
      overlap_dof_handlers;

    /**
     * Translations between overlap dofs and native dofs.
     */
    std::vector<std::vector<types::global_dof_index>>
      overlap_to_native_dof_translations;

    /**
     * Scatter objects for moving vectors between native and overlap
     * representations. Indexed first by the number of the dof handler.
     */
    std::vector<std::vector<Scatter<double>>> scatters;
    /**
     * @}
     */
  };
} // namespace fdl
#endif
