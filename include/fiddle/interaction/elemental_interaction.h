#ifndef included_fiddle_interaction_elemental_interaction_h
#define included_fiddle_interaction_elemental_interaction_h

#include <fiddle/base/quadrature_family.h>

#include <fiddle/interaction/interaction_base.h>

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
     * Constructor. Sets up an empty object.
     */
    ElementalInteraction(const unsigned int min_n_points_1D,
                         const double       point_density);

    /**
     * Constructor.
     */
    ElementalInteraction(
      const parallel::shared::Triangulation<dim, spacedim> &native_tria,
      const std::vector<BoundingBox<spacedim, float>> &     active_cell_bboxes,
      const std::vector<float> &                            active_cell_lengths,
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
      const int                                             level_number,
      const unsigned int                                    min_n_points_1D,
      const double                                          point_density);

    /**
     * Reinitialize the object. Same as the constructor, except min_n_points_1D
     * and point_density are unchanged.
     */
    virtual void
    reinit(const parallel::shared::Triangulation<dim, spacedim> &native_tria,
           const std::vector<BoundingBox<spacedim, float>> &active_cell_bboxes,
           const std::vector<float> &                       active_cell_lengths,
           tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
           const int level_number) override;

    /**
     * Middle part of velocity interpolation - performs the actual
     * computations and does not communicate.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_projection_rhs_intermediate(
      std::unique_ptr<TransactionBase> transaction) const override;

    /**
     * Middle part of force spreading - performs the actual computations and
     * does not communicate.
     */
    virtual std::unique_ptr<TransactionBase>
    compute_spread_intermediate(
      std::unique_ptr<TransactionBase> transaction) override;

    virtual std::unique_ptr<TransactionBase>
    add_workload_start(
      const int                                         workload_index,
      const LinearAlgebra::distributed::Vector<double> &position,
      const DoFHandler<dim, spacedim> &position_dof_handler) override;

    virtual std::unique_ptr<TransactionBase>
    add_workload_intermediate(std::unique_ptr<TransactionBase> t_ptr) override;

    virtual void
    add_workload_finish(std::unique_ptr<TransactionBase> t_ptr) override;

  protected:
    unsigned int min_n_points_1D;

    double point_density;

    /**
     * Indices of the quadrature rules that should be used on each cell.
     */
    std::vector<unsigned char> quadrature_indices;

    /**
     * Collection of quadrature rules which are suitable for the given
     * triangulation.
     */
    std::unique_ptr<QuadratureFamily<dim>> quadrature_family;

    /**
     * Vector of quadratures we will actually use for interaction.
     */
    std::vector<Quadrature<dim>> quadratures;
  };
} // namespace fdl
#endif
