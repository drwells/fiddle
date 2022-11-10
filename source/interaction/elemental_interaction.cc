#include <fiddle/base/samrai_utilities.h>

#include <fiddle/interaction/elemental_interaction.h>
#include <fiddle/interaction/interaction_utilities.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_tools.h>

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
    const double       point_density,
    const DensityKind  density_kind)
    : InteractionBase<dim, spacedim>()
    , min_n_points_1D(min_n_points_1D)
    , point_density(point_density)
    , density_kind(density_kind)
  {}

  template <int dim, int spacedim>
  ElementalInteraction<dim, spacedim>::ElementalInteraction(
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
    const std::vector<float>                             &active_cell_lengths,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
    const int                                             level_number,
    const unsigned int                                    min_n_points_1D,
    const double                                          point_density,
    const DensityKind                                     density_kind)
    : ElementalInteraction<dim, spacedim>(min_n_points_1D,
                                          point_density,
                                          density_kind)
  {
    reinit(native_tria,
           active_cell_bboxes,
           active_cell_lengths,
           patch_hierarchy,
           level_number);
  }

  template <int dim, int spacedim>
  void
  ElementalInteraction<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>>  &global_active_cell_bboxes,
    const std::vector<float>                         &active_cell_lengths,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
    const int                                         level_number)
  {
    InteractionBase<dim, spacedim>::reinit(native_tria,
                                           global_active_cell_bboxes,
                                           active_cell_lengths,
                                           patch_hierarchy,
                                           level_number);
    const auto patches =
      extract_patches(patch_hierarchy->getPatchLevel(level_number));

    // Set up the patch map:
    {
      // Yes, this is much more complex than necessary since
      // global_active_cell_bboxes is an argument to this function, but we don't
      // want to rely on that and p::s::T more than we have to since that
      // approach ultimately needs to go.
      //
      // TODO - we should refactor this into a more general function so we can
      // test it
      std::vector<CellId> bbox_cellids;
      for (const auto &cell : this->overlap_tria.active_cell_iterators())
        bbox_cellids.push_back(this->overlap_tria.get_native_cell_id(cell));

      // 1. Figure out who owns the bounding boxes we need:
      std::vector<types::subdomain_id> ranks =
        GridTools::get_subdomain_association(*this->native_tria, bbox_cellids);

      // 2. Send each processor the list of bboxes we need:
      std::map<types::subdomain_id,
               std::vector<std::pair<unsigned int, CellId>>>
        corresponding_requested_cellids;
      // Keep the overlap active cell index along for the ride
      for (unsigned int i = 0; i < ranks.size(); ++i)
        corresponding_requested_cellids[ranks[i]].emplace_back(i,
                                                               bbox_cellids[i]);

      const std::map<types::subdomain_id,
                     std::vector<std::pair<unsigned int, CellId>>>
        corresponding_cellids_to_send =
          Utilities::MPI::some_to_some(this->communicator,
                                       corresponding_requested_cellids);

      // 3. Send each processor the actual bboxes:
      std::map<
        types::subdomain_id,
        std::vector<std::pair<unsigned int, BoundingBox<spacedim, float>>>>
        requested_bboxes;
      for (const auto &pair : corresponding_cellids_to_send)
        {
          const auto  rank                = pair.first;
          const auto &indices_and_cellids = pair.second;

          auto &bboxes = requested_bboxes[rank];
          for (const auto &index_and_cellid : indices_and_cellids)
            {
              auto it = this->native_tria->create_cell_iterator(
                index_and_cellid.second);
              bboxes.emplace_back(
                index_and_cellid.first,
                global_active_cell_bboxes[it->active_cell_index()]);
            }
        }

      const auto received_bboxes =
        Utilities::MPI::some_to_some(this->communicator, requested_bboxes);

      std::vector<BoundingBox<spacedim, float>> overlap_bboxes(
        this->overlap_tria.n_active_cells());
      for (const auto &pair : received_bboxes)
        for (const auto &index_and_bbox : pair.second)
          {
            AssertIndexRange(index_and_bbox.first, overlap_bboxes.size());
            overlap_bboxes[index_and_bbox.first] = index_and_bbox.second;
          }

      // TODO add the ghost cell width as an input argument to this class
      patch_map.reinit(patches, 1.0, this->overlap_tria, overlap_bboxes);
    }

    // We need to implement some more quadrature families
    const auto reference_cells = native_tria.get_reference_cells();
    Assert(reference_cells.size() == 1, ExcFDLNotImplemented());
    if (!quadrature_family)
      {
        if (reference_cells.front() == ReferenceCells::get_hypercube<dim>())
          quadrature_family.reset(
            new QGaussFamily<dim>(min_n_points_1D, point_density));
        else if (reference_cells.front() == ReferenceCells::get_simplex<dim>())
          quadrature_family.reset(new QWitherdenVincentSimplexFamily<dim>(
            min_n_points_1D, point_density, density_kind));
        else
          Assert(false, ExcFDLNotImplemented());
      }

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
        const auto   native_cell = this->overlap_tria.get_native_cell(cell);
        const double lagrangian_length =
          active_cell_lengths[native_cell->active_cell_index()];
        quadrature_indices.push_back(
          quadrature_family->get_index(eulerian_length, lagrangian_length));
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
  bool
  ElementalInteraction<dim, spacedim>::projection_is_interpolation() const
  {
    return false;
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  ElementalInteraction<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // Finish communication:
    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    MappingFEField<dim, spacedim, Vector<double>> position_mapping(
      this->get_overlap_dof_handler(*trans.native_position_dof_handler),
      trans.overlap_position);

    // Actually do the interpolation:
    compute_projection_rhs(trans.kernel_name,
                           trans.current_data_idx,
                           patch_map,
                           position_mapping,
                           quadrature_indices,
                           quadratures,
                           this->get_overlap_dof_handler(
                             *trans.native_dof_handler),
                           *trans.mapping,
                           trans.overlap_rhs);

    // After we compute we begin the scatter back to the native partitioning:
    trans.rhs_scatter.overlap_to_global_start(trans.overlap_rhs,
                                              trans.rhs_scatter_back_op,
                                              0,
                                              *trans.native_rhs);

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
    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    trans.solution_scatter.global_to_overlap_finish(*trans.native_solution,
                                                    trans.overlap_solution);

    MappingFEField<dim, spacedim, Vector<double>> position_mapping(
      this->get_overlap_dof_handler(*trans.native_position_dof_handler),
      trans.overlap_position);

    // Actually do the spreading:
    compute_spread(trans.kernel_name,
                   trans.current_data_idx,
                   patch_map,
                   position_mapping,
                   quadrature_indices,
                   quadratures,
                   this->get_overlap_dof_handler(*trans.native_dof_handler),
                   *trans.mapping,
                   trans.overlap_solution);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  ElementalInteraction<dim, spacedim>::add_workload_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<WorkloadTransaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.next_state ==
            WorkloadTransaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // Finish communication:
    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    MappingFEField<dim, spacedim, Vector<double>> position_mapping(
      this->get_overlap_dof_handler(*trans.native_position_dof_handler),
      trans.overlap_position);

    count_quadrature_points(trans.workload_index,
                            patch_map,
                            position_mapping,
                            quadrature_indices,
                            quadratures);

    trans.next_state = WorkloadTransaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  VectorOperation::values
  ElementalInteraction<dim, spacedim>::get_rhs_scatter_type() const
  {
    return VectorOperation::add;
  }



  // instantiations
  template class ElementalInteraction<NDIM - 1, NDIM>;
  template class ElementalInteraction<NDIM, NDIM>;
} // namespace fdl
