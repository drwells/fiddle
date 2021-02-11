#ifndef included_fiddle_interaction_interaction_h
#define included_fiddle_interaction_interaction_h

#include <fiddle/base/quadrature_family.h>

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <ibtk/SAMRAIDataCache.h>
#include <ibtk/SAMRAIGhostDataAccumulator.h>

#include <PatchLevel.h>

#include <memory>
#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(const std::vector<BoundingBox<spacedim, Number>> &bboxes,
            const int                                         tag_index,
            tbox::Pointer<hier::PatchLevel<spacedim>>         patch_level);

  /**
   * Compute the right-hand side used to project the velocity from Eulerian to
   * Lagrangian representation.
   *
   * @param[in] data_idx the SAMRAI patch data index we are interpolating. The
   * depth of the variable must match the number of components of the finite
   * element.
   *
   * @param[in] patch_map The mapping between SAMRAI patches and deal.II cells
   * which we will use for interpolation.
   *
   * @param[in] position_mapping Mapping from the reference configuration to the
   * current configuration of the mesh.
   *
   * @param[in] quadrature_indices This vector is indexed by the active cell
   * index - the value is the index into @p quadratures corresponding to the
   * correct quadrature rule on that cell.
   *
   * @param[in] f_dof_handler DoFHandler for the finite element we are
   * interpolating onto.
   *
   * @param[in] f_mapping Mapping for computing values of the finite element
   * field on the reference configuration.
   *
   * @param[out] f_rhs The load vector populated by this operation.
   *
   * @note In general, an OverlappingTriangulation has no knowledge of whether
   * or not DoFs on its boundaries should be constrained. Hence information must
   * first be communicated between processes and then constraints should be
   * applied.
   */
  template <int dim, int spacedim = dim>
  void
  compute_projection_rhs(const int                           f_data_idx,
                         const PatchMap<dim, spacedim> &     patch_map,
                         const Mapping<dim, spacedim> &      position_mapping,
                         const std::vector<unsigned char> &  quadrature_indices,
                         const std::vector<Quadrature<dim>> &quadratures,
                         const DoFHandler<dim, spacedim> &   f_dof_handler,
                         const Mapping<dim, spacedim> &      f_mapping,
                         Vector<double> &                    f_rhs);

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
  {};

  /**
   * Standard class for transactions - used by ElementalInteraction and
   * NodalInteraction.
   */
  template <int dim, int spacedim = dim>
  struct Transaction : public TransactionBase
  {
    /// Current patch index.
    int current_f_data_idx;

    /// Quadrature family.
    SmartPointer<QuadratureFamily<dim>> quad_family;

    /// Quadrature indices (temporarily casted to floats - fix this later)
    std::unique_ptr<Vector<float>> quad_indices;

    /// Native position DoFHandler.
    SmartPointer<const DoFHandler<dim, spacedim>> native_X_dof_handler;

    /// Native-partitioned position.
    SmartPointer<const LinearAlgebra::distributed::Vector<double>> native_X;

    /// Overlap-partitioned position.
    std::unique_ptr<Vector<double>> overlap_X_vec;

    /// Native F DoFHandler.
    SmartPointer<const DoFHandler<dim, spacedim>> native_F_dof_handler;

    /// Native-partitioned F_rhs.
    SmartPointer<LinearAlgebra::distributed::Vector<double>> native_F_rhs;

    /// Overlap-partitioned F.
    std::unique_ptr<Vector<double>> overlap_F_vec;

    /// Mapping to use for F.
    SmartPointer<const Mapping<dim, spacedim>> F_mapping;

    /// Possible states for a transaction.
    enum class State
    {
      Start,
      Intermediate,
      Finish
    };

    /// Next state. Used for consistency checking.
    State next_state;

    /// Possible operations.
    enum class Operation
    {
    interpolation,
    spreading
    };

    /// Operation of the current transaction. Used for consistency checking.
    Operation operation;
  };

  /**
   * Base class managing interaction between SAMRAI and deal.II data structures,
   * by interpolation and spreading, where the position of the structure is
   * described by a finite element field. This class sets up the data structures
   * and communication patterns necessary for all types of interaction (like
   * nodal or elemental coupling).
   */
  template <int dim, int spacedim = dim>
  class InteractionBase
  {
  public:
    /**
     * Constructor. This call is collective.
     *
     * @param[in] native_tria The Triangulation used to define the finite
     *            element fields. This class will use the same MPI communicator
     *            as the one used by this Triangulation.
     *
     * @param[in] active_cell_bboxes Bounding box for each active cell on the
     *            current processor. This should be computed with the finite
     *            element description of the displacement.
     *
     * @param[inout] patch_hierarchy The patch hierarchy with which we will
     *               interact (i.e., for spreading and interpolation).
     *
     * @param[in] level_number Number of the level on which we are interacting.
     *            Multilevel IBFE is not yet supported.
     *
     * @param[inout] eulerian_data_cache Pointer to the shared cache of
     *               scratch patch indices of @p patch_hierarchy.
     */
    InteractionBase(
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>> &     active_cell_bboxes,
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
      const int                                             level_number,
      std::shared_ptr<IBTK::SAMRAIDataCache> eulerian_data_cache);

    /**
     * Reinitialize the object. Same as the constructor.
     */
    virtual void
    reinit(
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>> &     active_cell_bboxes,
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
      const int                                             level_number,
      std::shared_ptr<IBTK::SAMRAIDataCache> eulerian_data_cache = {});

    /**
     * Store a pointer to @p native_dof_handler and also compute the
     * equivalent DoFHandler on the overlapping partitioning.
     *
     * This call is collective over the communicator used by this class.
     */
    virtual void
    add_dof_handler(const DoFHandler<dim, spacedim> &native_dof_handler);

    /**
     * Start the computation of the RHS vector corresponding to projecting @p
     * f_data_idx onto the finite element space specified by @p F_dof_handler.
     * Since interpolation requires multiple data transfers it is split into
     * three parts. In particular, this first function begins the asynchronous
     * scatter from the native representation to the overlapping
     * representation.
     *
     * @return This function returns a Transaction object which completely
     * encapsulates the current state of the interpolation.
     *
     * @warning The Transaction returned by this method stores pointers to all
     * of the input arguments. Those pointers must remain valid until after
     * compute_projection_rhs_finish is called.
     */
    virtual
    std::unique_ptr<TransactionBase>
    compute_projection_rhs_start(
      const int                                         f_data_idx,
      const QuadratureFamily<dim> &                     quad_family,
      const std::vector<unsigned char> &                quad_indices,
      const DoFHandler<dim, spacedim> &                 X_dof_handler,
      const LinearAlgebra::distributed::Vector<double> &X,
      const DoFHandler<dim, spacedim> &                 F_dof_handler,
      const Mapping<dim, spacedim> &                    F_mapping,
      LinearAlgebra::distributed::Vector<double> &      F_rhs) const;


    /**
     * Middle part of velocity interpolation - performs the actual
     * computations and does not communicate.
     */
    virtual
    std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const = 0;

    /**
     * Finish the computation of the RHS vector corresponding to projecting @p
     * f_data_idx onto the finite element space specified by @p F_dof_handler.
     * This step accumulates the RHS vector computed in the overlap
     * representation back to the native representation.
     */
    virtual void
    compute_projection_rhs_finish(
      std::unique_ptr<TransactionBase> transaction) const;

#if 0
    /**
     * Start spreading forces from the provided finite element field @p F by
     * adding them onto the SAMRAI data index @p f_data_idx.
     *
     * @warning The Transaction returned by this method stores pointers to all
     * of the input arguments. Those pointers must remain valid until after
     * compute_projection_rhs_finish is called.
     */
    virtual
    std::unique_ptr<TransactionBase>
    spread_force_start(const int                              f_data_idx,
                       const QuadratureFamily<dim, spacedim> &quad_family,
                       const std::vector<unsigned char> &     quad_indices,
                       const LinearAlgebra::distributed::Vector<double> &X,
                       const DoFHandler<dim, spacedim> &X_dof_handler,
                       const Mapping<dim, spacedim> &   F_mapping,
                       const DoFHandler<dim, spacedim> &F_dof_handler,
                       // TODO - we need something that can accumulate forces
                       // spread outside the domain in spread_force_finish
                       const LinearAlgebra::distributed::Vector<double> &F);

    /**
     * Middle part of force spreading - performs the actual computations and
     * does not communicate.
     */
    virtual
    std::unique_ptr<TransactionBase>
    spread_force_intermediate(std::unique_ptr<TransactionBase> spread_transaction);

    /**
     * Finish spreading forces from the provided finite element field @p F by
     * adding them onto the SAMRAI data index @p f_data_idx.
     */
    virtual
    std::unique_ptr<TransactionBase>
    spread_force_finish(std::unique_ptr<TransactionBase> spread_transaction);
#endif

  protected:
    /**
     * @name Geometric data.
     * @{
     */

    /**
     * Native triangulation, which is stored separately.
     */
    SmartPointer<parallel::shared::Triangulation<dim, spacedim>> native_tria;

    /**
     * Overlap triangulation - i.e., the part of native_tria that intersects the
     * patches in patch_level stored on the current processor.
     */
    OverlapTriangulation<dim, spacedim> overlap_tria;

    /**
     * Mapping from SAMRAI patches to deal.II cells.
     */
    PatchMap<dim, spacedim> patch_map;

    /**
     * Pointer to the patch hierarchy.
     */
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy;

    /**
     * Number of the patch level we interact with.
     */
    int level_number;

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
    std::vector<SmartPointer<const DoFHandler<dim, spacedim>>> native_dof_handlers;

    /**
     * DoFHandlers defined on the overlap tria, which are equivalent to those
     * stored by @p native_dof_handlers.
     */
    std::vector<DoFHandler<dim, spacedim>> overlap_dof_handlers;

    /**
     * Scatter objects for moving vectors between native and overlap
     * representations.
     */
    std::vector<Scatter<double>> scatters;

    /**
     * Pointer to the (possibly shared) cache of Eulerian data (i.e., the object
     * that keeps track of scratch patch indices).
     *
     * TODO finish this when we implement spreading
     */
    std::shared_ptr<IBTK::SAMRAIDataCache> eulerian_data_cache;

    /**
     * TODO finish this when we implement spreading
     */
    std::unique_ptr<IBTK::SAMRAIGhostDataAccumulator> ghost_data_accumulator;

    /**
     * @}
     */

    /**
     * @name Data structures used for internal communication.
     * @{
     */

    /**
     * Scatter object for moving cell data (computed as active cell indices).
     *
     * @todo There is probably a more efficient way to implement this.
     */
    Scatter<float> cell_index_scatter;

    /**
     * @}
     */
  };

  template <int dim, int spacedim = dim>
  class ElementalInteraction : public InteractionBase<dim, spacedim>
  {
  public:
    /**
     * Reuse the base class constructor.
     */
    using InteractionBase<dim, spacedim>::InteractionBase;

    /**
     * Middle part of velocity interpolation - performs the actual
     * computations and does not communicate.
     */
    virtual
    std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const override;

#if 0
    /**
     * Middle part of force spreading - performs the actual computations and
     * does not communicate.
     */
    virtual void
    spread_force_intermediate(SpreadTransaction &spread_transaction) override;

    /**
     * Finish spreading forces from the provided finite element field @p F by
     * adding them onto the SAMRAI data index @p f_data_idx.
     */
    virtual void
    spread_force_finish(SpreadTransaction &spread_transaction) override;
#endif
  };


  template <int dim, int spacedim = dim>
  class NodalInteraction : public InteractionBase<dim, spacedim>
  {
  public:
    /**
     * TODO copydoc
     */
    NodalInteraction(
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>>       active_cell_bboxes,
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
      const int                                             level_number,
      std::shared_ptr<IBTK::SAMRAIDataCache> eulerian_data_cache = {});

    /**
     * TODO
     */
    virtual void
    reinit(
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_patch_hierarchy,
      const int                                         level_number,
      const IntersectionPredicate<dim, spacedim> &      predicate) override;

    /**
     * TODO
     */
    virtual
    std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const override;

#if 0
    /**
     * TODO
     */
    virtual void
    spread_force_intermediate(SpreadTransaction &spread_transaction) override;

    /**
     * TODO
     */
    virtual void
    spread_force_finish(SpreadTransaction &spread_transaction) override;
#endif
  };
} // namespace fdl
#endif
