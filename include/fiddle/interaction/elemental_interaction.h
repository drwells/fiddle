#ifndef included_fiddle_interaction_elemental_interaction_h
#define included_fiddle_interaction_elemental_interaction_h

#include <fiddle/base/quadrature_family.h>

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

#include <memory>
#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

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
    virtual std::unique_ptr<TransactionBase>
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
} // namespace fdl
#endif
