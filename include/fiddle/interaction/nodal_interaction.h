#ifndef included_fiddle_interaction_nodal_interaction_h
#define included_fiddle_interaction_nodal_interaction_h

#include <fiddle/base/config.h>

#include <fiddle/grid/nodal_patch_map.h>

#include <fiddle/interaction/interaction_base.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <memory>
#include <vector>

// forward declarations
namespace dealii
{
  template <int, int>
  class Mapping;

  namespace parallel
  {
    namespace shared
    {
      template <int, int>
      class Triangulation;
    }
  } // namespace parallel
} // namespace dealii

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class PatchHierarchy;
  }

  namespace tbox
  {
    template <typename>
    class Pointer;
  }
} // namespace SAMRAI

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim = dim>
  class NodalInteraction : public InteractionBase<dim, spacedim>
  {
  public:
    /**
     * Default constructor. Sets up an empty object.
     */
    NodalInteraction();

    /**
     * Constructor.
     */
    NodalInteraction(
      const tbox::Pointer<tbox::Database>                  &input_db,
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
      tbox::Pointer<hier::PatchHierarchy<spacedim>>         patch_hierarchy,
      const std::pair<int, int>                            &level_numbers,
      const DoFHandler<dim, spacedim>                  &position_dof_handler,
      const LinearAlgebra::distributed::Vector<double> &position);

    /**
     * Override of the base-class version. This function doesn't make any sense
     * for the present class since it depends on nodal data - hence, to prevent
     * programming errors, this method just throws an exception should it be
     * called.
     */
    virtual void
    reinit(const tbox::Pointer<tbox::Database>                  &input_db,
           const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>> &active_cell_bboxes,
           const std::vector<float>                        &active_cell_lengths,
           tbox::Pointer<hier::PatchHierarchy<spacedim>>    patch_hierarchy,
           const std::pair<int, int> &level_numbers) override;

    /**
     * Reinitialize the object. Same as the constructor.
     */
    virtual void
    reinit(const tbox::Pointer<tbox::Database>                  &input_db,
           const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>> &active_cell_bboxes,
           tbox::Pointer<hier::PatchHierarchy<spacedim>>    patch_hierarchy,
           const std::pair<int, int>                       &level_numbers,
           const DoFHandler<dim, spacedim> &position_dof_handler,
           const LinearAlgebra::distributed::Vector<double> &position);

    /**
     * Same as base class but also sets up some necessary internal data
     * structures used by this class
     */
    virtual void
    add_dof_handler(
      const DoFHandler<dim, spacedim> &native_dof_handler) override;

    /**
     * This method always interpolates so this always returns true.
     */
    virtual bool
    projection_is_interpolation() const override;

    /**
     * Do the actual work associated with nodal interpolation by, if necessary,
     * computing nodes and then calling compute_nodal_interpolation().
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const override;

    /**
     * Finish nodal interaction. Unlike the base class method this method sets
     * velocities of nodes outside the domain to zero.
     */
    virtual void
    compute_projection_rhs_accumulate_finish(
      std::unique_ptr<TransactionBase> transaction) override;

    /**
     * Convenience method which calls all of the other interpolation functions
     * in the correct order. Some care should be taken when using this function
     * since it may exhibit poor performance with multiple parts.
     */
    void
    interpolate(const std::string               &kernel_name,
                const int                        data_idx,
                const DoFHandler<dim, spacedim> &position_dof_handler,
                const LinearAlgebra::distributed::Vector<double> &position,
                const DoFHandler<dim, spacedim>                  &dof_handler,
                const Mapping<dim, spacedim>                     &mapping,
                LinearAlgebra::distributed::Vector<double>       &result);

    /**
     * Do some checks and then call compute_nodal_spread().
     */
    virtual std::unique_ptr<TransactionBase>
    compute_spread_intermediate(
      std::unique_ptr<TransactionBase> transaction) override;

    /**
     * Middle part of communicating workload. Does not communicate.
     */
    virtual std::unique_ptr<TransactionBase>
    add_workload_intermediate(std::unique_ptr<TransactionBase> t_ptr) override;

  protected:
    virtual VectorOperation::values
    get_rhs_scatter_type() const override;

    const NodalPatchMap<dim, spacedim> &
    get_nodal_patch_map(
      const DoFHandler<dim, spacedim> &native_dof_handler) const;

    /**
     * For convenience, store an explicit pointer to the natively partitioned
     * position DoFHandler (the base class also stores a pointer).
     */
    SmartPointer<const DoFHandler<dim, spacedim>> native_position_dof_handler;

    /**
     * Similarly, to construct NodalPatchMaps, we need the displacement field
     * on the OverlapTriangulation, so store that vector here:
     */
    Vector<double> overlap_position;

    /**
     * Patches used for interaction.
     */
    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;

    /**
     * Bounding boxes for each patch.
     */
    std::vector<std::vector<BoundingBox<spacedim>>> bboxes;

    /**
     * Mappings between support points (nodes) and patches. Indexed by the
     * number of the DoFHandler.
     *
     * @note These are filled at first use, so the container is mutable.
     */
    mutable std::vector<std::shared_ptr<NodalPatchMap<dim, spacedim>>>
      nodal_patch_maps;
  };
} // namespace fdl
#endif
