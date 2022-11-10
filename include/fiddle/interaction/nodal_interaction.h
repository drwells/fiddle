#ifndef included_fiddle_interaction_nodal_interaction_h
#define included_fiddle_interaction_nodal_interaction_h

#include <fiddle/base/config.h>

#include <fiddle/base/quadrature_family.h>

#include <fiddle/grid/nodal_patch_map.h>
#include <fiddle/grid/overlap_tria.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <BasePatchHierarchy.h>

#include <memory>
#include <vector>

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
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
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
    reinit(const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>> &active_cell_bboxes,
           const std::vector<float>                        &active_cell_lengths,
           tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
           const std::pair<int, int> &level_numbers) override;

    /**
     * Reinitialize the object. Same as the constructor.
     */
    virtual void
    reinit(const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>>  &active_cell_bboxes,
           tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
           const std::pair<int, int>                        &level_numbers,
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
     * TODO
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const override;

    /**
     * TODO
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

    NodalPatchMap<dim, spacedim> nodal_patch_map;
  };
} // namespace fdl
#endif
