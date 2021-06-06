#include <fiddle/base/samrai_utilities.h>

#include <fiddle/interaction/elemental_interaction.h>
#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/base/mpi.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <CartesianPatchGeometry.h>

#include <cmath>
#include <numeric>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim>
  ElementalInteraction<dim, spacedim>::ElementalInteraction(
    const unsigned int min_n_points_1D,
    const double       point_density)
    : InteractionBase<dim, spacedim>()
    , min_n_points_1D(min_n_points_1D)
    , point_density(point_density)
  {}

  template <int dim, int spacedim>
  ElementalInteraction<dim, spacedim>::ElementalInteraction(
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>> &     active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
    const int                                             level_number,
    const unsigned int                                    min_n_points_1D,
    const double                                          point_density)
    : ElementalInteraction<dim, spacedim>(min_n_points_1D, point_density)
  {
    reinit(native_tria, active_cell_bboxes, patch_hierarchy, level_number);
  }

  template <int dim, int spacedim>
  void
  ElementalInteraction<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>> &     active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
    const int                                             level_number)
  {
    InteractionBase<dim, spacedim>::reinit(native_tria,
                                           active_cell_bboxes,
                                           patch_hierarchy,
                                           level_number);
    // We need to implement some more quadrature families
    Assert(native_tria.all_reference_cells_are_hyper_cube(),
           ExcFDLNotImplemented());
    quadrature_family.reset(
      new QGaussFamily<dim>(min_n_points_1D, point_density));

    const auto patches =
      extract_patches(patch_hierarchy->getPatchLevel(level_number));
    double patch_dx_min = std::numeric_limits<double>::max();
    if (patches.size() > 0)
      {
        const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geometry =
          patches[0]->getPatchGeometry();
        const double *const patch_dx = geometry->getDx();
        patch_dx_min = *std::min_element(patch_dx, patch_dx + spacedim);
      }
    const double eulerian_length =
      Utilities::MPI::min(patch_dx_min, this->communicator);

    // Determine which quadrature rule we should use on each cell:
    quadrature_indices.resize(0);
    for (const auto &cell : this->overlap_tria.active_cell_iterators())
      {
        const auto  native_cell = this->overlap_tria.get_native_cell(cell);
        const auto &native_bbox =
          active_cell_bboxes[native_cell->active_cell_index()];

        double lagrangian_length = native_bbox.side_length(0);
        for (unsigned int d = 1; d < spacedim; ++d)
          lagrangian_length =
            std::max<double>(lagrangian_length, native_bbox.side_length(d));

        quadrature_indices.push_back(
          quadrature_family->get_n_points_1D(eulerian_length,
                                             lagrangian_length));
      }

    // Store quadratures in a vector:
    unsigned char max_quadrature_index = 0;
    if (quadrature_indices.size() > 0)
      max_quadrature_index =
        *std::max_element(quadrature_indices.begin(), quadrature_indices.end());
    quadratures.resize(0);
    for (unsigned char i = 0; i <= max_quadrature_index; ++i)
      quadratures.push_back((*quadrature_family)[i]);
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  ElementalInteraction<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // Finish communication:
    trans.X_scatter.global_to_overlap_finish(*trans.native_X,
                                             trans.overlap_X_vec);

    MappingFEField<dim, spacedim, Vector<double>> X_mapping(
      this->get_overlap_dof_handler(*trans.native_X_dof_handler),
      trans.overlap_X_vec);

    // Actually do the interpolation:
    compute_projection_rhs(trans.current_f_data_idx,
                           this->patch_map,
                           X_mapping,
                           quadrature_indices,
                           quadratures,
                           this->get_overlap_dof_handler(
                             *trans.native_F_dof_handler),
                           *trans.F_mapping,
                           trans.overlap_F);

    // After we compute we begin the scatter back to the native partitioning:
    trans.F_scatter.overlap_to_global_start(trans.overlap_F,
                                            VectorOperation::add,
                                            0,
                                            *trans.native_F_rhs);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  ElementalInteraction<dim, spacedim>::compute_spread_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // Finish communication:
    trans.X_scatter.global_to_overlap_finish(*trans.native_X,
                                             trans.overlap_X_vec);

    trans.F_scatter.global_to_overlap_finish(*trans.native_F, trans.overlap_F);

    MappingFEField<dim, spacedim, Vector<double>> X_mapping(
      this->get_overlap_dof_handler(*trans.native_X_dof_handler),
      trans.overlap_X_vec);

    // Actually do the spreading:
    compute_spread(trans.current_f_data_idx,
                   this->patch_map,
                   X_mapping,
                   quadrature_indices,
                   quadratures,
                   this->get_overlap_dof_handler(*trans.native_F_dof_handler),
                   *trans.F_mapping,
                   trans.overlap_F);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }

  // instantiations
  template class ElementalInteraction<NDIM - 1, NDIM>;
  template class ElementalInteraction<NDIM, NDIM>;
} // namespace fdl
