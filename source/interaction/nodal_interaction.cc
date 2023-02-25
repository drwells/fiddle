#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/interaction_utilities.h>
#include <fiddle/interaction/nodal_interaction.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <CartesianPatchGeometry.h>

#include <cmath>
#include <numeric>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  namespace
  {
    // Helper function for computing the locations of nodes for DoFHandlers
    // with different base elements from the position DoFHandler.
    template <int dim, int spacedim>
    Vector<double>
    compute_nodes(const DoFHandler<dim, spacedim> &position_dof_handler,
                  const Vector<double>            &position,
                  const DoFHandler<dim, spacedim> &dof_handler)
    {
      MappingFEField<dim, spacedim, Vector<double>> position_mapping(
        position_dof_handler, position);
      std::vector<Point<spacedim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(position_mapping,
                                           dof_handler,
                                           support_points);
      support_points.erase(std::unique(support_points.begin(),
                                       support_points.end()),
                           support_points.end());
      Assert(dof_handler.n_dofs() ==
               support_points.size() * dof_handler.get_fe().n_components(),
             ExcFDLInternalError());

      Vector<double> nodal_coordinates(support_points.size() * spacedim);
      for (unsigned int node_n = 0; node_n < support_points.size(); ++node_n)
        for (unsigned int d = 0; d < spacedim; ++d)
          nodal_coordinates[node_n * spacedim + d] = support_points[node_n][d];

      return nodal_coordinates;
    }
  } // namespace

  template <int dim, int spacedim>
  NodalInteraction<dim, spacedim>::NodalInteraction()
  {}

  template <int dim, int spacedim>
  NodalInteraction<dim, spacedim>::NodalInteraction(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
    const std::pair<int, int>                            &level_numbers,
    const DoFHandler<dim, spacedim>                      &position_dof_handler,
    const LinearAlgebra::distributed::Vector<double>     &position)
  {
    reinit(input_db,
           native_tria,
           active_cell_bboxes,
           patch_hierarchy,
           level_numbers,
           position_dof_handler,
           position);
  }

  template <int dim, int spacedim>
  void
  NodalInteraction<dim, spacedim>::reinit(
    const tbox::Pointer<tbox::Database> &/*input_db*/,
    const parallel::shared::Triangulation<dim, spacedim> & /*native_tria*/,
    const std::vector<BoundingBox<spacedim, float>> & /*active_cell_bboxes*/,
    const std::vector<float> & /*active_cell_lengths*/,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> /*patch_hierarchy*/,
    const std::pair<int, int> & /*level_numbers*/)
  {
    AssertThrow(false,
                ExcMessage(
                  "This version of reinit cannot be used with this class since "
                  "NodalInteraction requires the nodal coordinate vector to do "
                  "reinitialization."));
  }

  template <int dim, int spacedim>
  void
  NodalInteraction<dim, spacedim>::reinit(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &native_tria,
    const std::vector<BoundingBox<spacedim, float>>      &active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     patch_hierarchy,
    const std::pair<int, int>                            &level_numbers,
    const DoFHandler<dim, spacedim>                      &position_dof_handler,
    const LinearAlgebra::distributed::Vector<double>     &position)
  {
    // base class doesn't actually read this value
    std::vector<float> active_cell_lengths;
    InteractionBase<dim, spacedim>::reinit(input_db,
                                           native_tria,
                                           active_cell_bboxes,
                                           active_cell_lengths,
                                           patch_hierarchy,
                                           level_numbers);
    add_dof_handler(position_dof_handler);
    Vector<double> overlap_position(
      this->get_overlap_dof_handler(position_dof_handler).n_dofs());

    {
      Scatter<double> scatter = this->get_scatter(position_dof_handler);
      scatter.global_to_overlap_start(position, 0, overlap_position);
      scatter.global_to_overlap_finish(position, overlap_position);
      this->return_scatter(position_dof_handler, std::move(scatter));
    }

    const double ghost_cell_fraction =
      input_db->getDoubleWithDefault("ghost_cell_fraction", 1.0);

    // This won't work correctly yet with no ghost cell fraction
    AssertThrow(ghost_cell_fraction > 0.0, ExcFDLNotImplemented());

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;
    std::vector<std::vector<BoundingBox<spacedim>>>   bboxes;
    // add boxes which do not intersect the next finest patch level...
    for (int ln = level_numbers.first; ln < level_numbers.second; ++ln)
      {
        const auto patch_level       = patch_hierarchy->getPatchLevel(ln);
        const auto level_patches     = extract_patches(patch_level);
        const auto level_patch_boxes = compute_nonoverlapping_patch_boxes(
          patch_level, patch_hierarchy->getPatchLevel(ln + 1));
        AssertDimension(level_patches.size(), level_patch_boxes.size());

        patches.insert(patches.end(),
                       level_patches.begin(),
                       level_patches.end());
        for (const auto &patch_boxes : level_patch_boxes)
          {
            bboxes.emplace_back();
            for (const auto &box : patch_boxes)
              bboxes.back().push_back(box_to_bbox(box, patch_level));
          }
      }
    // ... and boxes on the finest patch level.
    const auto patch_level =
      patch_hierarchy->getPatchLevel(level_numbers.second);
    const auto level_patches =
      extract_patches(patch_hierarchy->getPatchLevel(level_numbers.second));
    patches.insert(patches.end(), level_patches.begin(), level_patches.end());
    for (const auto &patch : level_patches)
      {
        bboxes.emplace_back();
        bboxes.back().push_back(box_to_bbox(patch->getBox(), patch_level));
      }

    // Increase all the boxes by the ghost cell fraction:
    {
      double patch_dx_min = std::numeric_limits<double>::max();
      if (patches.size() > 0)
        {
          const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geometry =
            patches.back()->getPatchGeometry();
          Assert(geometry, ExcFDLNotImplemented());
          const double *const patch_dx = geometry->getDx();
          patch_dx_min = *std::min_element(patch_dx, patch_dx + spacedim);
        }
      for (auto &vec : bboxes)
        for (auto &box : vec)
          box.extend(patch_dx_min * ghost_cell_fraction);
    }

    // Set up class members:
    {
      nodal_patch_maps.resize(0);
      // position dof handler is always at index 0
      Assert(this->native_dof_handlers.size() == 1 &&
               &*this->native_dof_handlers[0] == &position_dof_handler,
             ExcFDLInternalError());
      nodal_patch_maps.push_back(std::make_shared<NodalPatchMap<dim, spacedim>>(
        patches, bboxes, overlap_position));

      this->native_position_dof_handler = &position_dof_handler;
      this->overlap_position            = std::move(overlap_position);
      this->patches                     = std::move(patches);
      this->bboxes                      = std::move(bboxes);
    }
  }

  template <int dim, int spacedim>
  void
  NodalInteraction<dim, spacedim>::add_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    // we only support vector-valued elements which have exactly one base
    // element
    AssertThrow(native_dof_handler.get_fe().n_base_elements() == 1,
                ExcFDLNotImplemented());

    // The rest is basically a cut-and-paste of the base class method, except
    // that we need to renumber the dofs halfway through.
    AssertThrow(&native_dof_handler.get_triangulation() == this->native_tria,
                ExcMessage("The DoFHandler must use the underlying native "
                           "triangulation."));
    const auto ptr = &native_dof_handler;
    if (std::find(this->native_dof_handlers.begin(),
                  this->native_dof_handlers.end(),
                  ptr) == this->native_dof_handlers.end())
      {
        this->native_dof_handlers.emplace_back(ptr);
        this->overlap_dof_handlers.emplace_back(
          std::make_unique<DoFHandler<dim, spacedim>>(this->overlap_tria));
        auto &overlap_dof_handler = *this->overlap_dof_handlers.back();
        overlap_dof_handler.distribute_dofs(
          native_dof_handler.get_fe_collection());

        // Here's the difficult part:
        // compute_overlap_to_native_dof_translation() needs the two DoFHandlers
        // to use the same numbering on each cell. Hence we have to call that
        // first and then combine it with the nodal renumbering.
        std::vector<types::global_dof_index> overlap_to_native_dofs =
          compute_overlap_to_native_dof_translation(this->overlap_tria,
                                                    overlap_dof_handler,
                                                    native_dof_handler);

        std::vector<types::global_dof_index> nodal_renumbering(
          overlap_dof_handler.n_dofs());
        DoFRenumbering::compute_support_point_wise(nodal_renumbering,
                                                   overlap_dof_handler);
        // if there is no renumbering then create one anyway - we still need it
        // for dof translation later
        if (nodal_renumbering.size() == 0)
          {
            nodal_renumbering.resize(overlap_dof_handler.n_dofs());
            std::iota(nodal_renumbering.begin(), nodal_renumbering.end(), 0u);
          }
        else
          {
            overlap_dof_handler.renumber_dofs(nodal_renumbering);
          }

        // Apply the permutation in reverse order:
        {
          for (std::size_t i = 0; i < overlap_to_native_dofs.size(); i++)
            {
              while (i != nodal_renumbering[i])
                {
                  const auto next = nodal_renumbering[i];
                  std::swap(overlap_to_native_dofs[i],
                            overlap_to_native_dofs[next]);
                  std::swap(nodal_renumbering[i], nodal_renumbering[next]);
                }
            }
        }

        this->overlap_to_native_dof_translations.emplace_back(
          std::move(overlap_to_native_dofs));
      }
  }

  template <int dim, int spacedim>
  bool
  NodalInteraction<dim, spacedim>::projection_is_interpolation() const
  {
    return true;
  }


  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  NodalInteraction<dim, spacedim>::compute_projection_rhs_intermediate(
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

    // If needed, convert the given position vector into the relevant nodal
    // one
    const bool reuse_nodes =
      trans.native_position_dof_handler->get_fe().base_element(0) ==
      trans.native_dof_handler->get_fe().base_element(0);
    Vector<double> nodal_coordinates;
    if (!reuse_nodes)
      {
        nodal_coordinates = compute_nodes(
          this->get_overlap_dof_handler(*trans.native_position_dof_handler),
          trans.overlap_position,
          this->get_overlap_dof_handler(*trans.native_dof_handler));
      }

    // Actually do the work:
    compute_nodal_interpolation(trans.kernel_name,
                                trans.current_data_idx,
                                get_nodal_patch_map(*trans.native_dof_handler),
                                reuse_nodes ? trans.overlap_position :
                                              nodal_coordinates,
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
  NodalInteraction<dim, spacedim>::compute_spread_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // It's not yet clear how to handle IB kernels with support on multiple
    // grid levels
    AssertThrow(this->level_numbers.first == this->level_numbers.second,
                ExcFDLNotImplemented());

    // Finish communication:
    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    trans.solution_scatter.global_to_overlap_finish(*trans.native_solution,
                                                    trans.overlap_solution);

    // Actually do the spreading:
    compute_nodal_spread(trans.kernel_name,
                         trans.current_data_idx,
                         // TODO: casting away const is bad
                         const_cast<NodalPatchMap<dim, spacedim> &>(
                           get_nodal_patch_map(*trans.native_dof_handler)),
                         trans.overlap_position,
                         trans.overlap_solution);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  NodalInteraction<dim, spacedim>::add_workload_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<WorkloadTransaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.next_state ==
            WorkloadTransaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // Finish communication:
    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    count_nodes(trans.workload_index,
                // TODO: casting away const is bad
                const_cast<NodalPatchMap<dim, spacedim> &>(
                  get_nodal_patch_map(*trans.native_position_dof_handler)),
                trans.overlap_position);

    trans.next_state = WorkloadTransaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }


  template <int dim, int spacedim>
  VectorOperation::values
  NodalInteraction<dim, spacedim>::get_rhs_scatter_type() const
  {
    return VectorOperation::max;
  }



  template <int dim, int spacedim>
  const NodalPatchMap<dim, spacedim> &
  NodalInteraction<dim, spacedim>::get_nodal_patch_map(
    const DoFHandler<dim, spacedim> &native_dof_handler) const
  {
    auto iter = std::find(this->native_dof_handlers.begin(),
                          this->native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != this->native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));

    const std::size_t index = iter - this->native_dof_handlers.begin();
    if (nodal_patch_maps.size() <= index)
      nodal_patch_maps.resize(index + 1);
    if (nodal_patch_maps[index] != nullptr)
      return *nodal_patch_maps[index];
    else if (native_dof_handler.get_fe().base_element(0) ==
             native_position_dof_handler->get_fe().base_element(0))
      {
        // if the base elements are the same, then we can reuse the position
        // patch map, regardless of the number of components
        nodal_patch_maps[index] = nodal_patch_maps[0];
      }
    else
      {
        // otherwise find the locations of the nodes and go from there
        const auto nodal_coordinates =
          compute_nodes(this->get_overlap_dof_handler(
                          *native_position_dof_handler),
                        overlap_position,
                        this->get_overlap_dof_handler(native_dof_handler));

        nodal_patch_maps[index] =
          std::make_shared<NodalPatchMap<dim, spacedim>>(patches,
                                                         bboxes,
                                                         nodal_coordinates);
      }
    return *nodal_patch_maps[index];
  }

  // instantiations
  template class NodalInteraction<NDIM - 1, NDIM>;
  template class NodalInteraction<NDIM, NDIM>;
} // namespace fdl
