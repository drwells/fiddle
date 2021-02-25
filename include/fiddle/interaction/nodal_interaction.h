#ifndef included_fiddle_interaction_nodal_interaction_h
#define included_fiddle_interaction_nodal_interaction_h

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <ibtk/SAMRAIDataCache.h>
#include <ibtk/SAMRAIGhostDataAccumulator.h>

#include <PatchLevel.h>
#include <fiddle/base/quadrature_family.h>

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
    virtual std::unique_ptr<TransactionBase>
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
