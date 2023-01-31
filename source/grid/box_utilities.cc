#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_values.h>

#include <CartesianGridGeometry.h>
#include <CartesianPatchGeometry.h>
#include <MultiblockPatchLevel.h>
#include <Patch.h>
#include <PatchLevel.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Extract the bounding boxes for a set of SAMRAI patches. In addition, if
   * necessary, expand each bounding box by @p extra_ghost_cell_fraction times
   * the length of a cell in each coordinate direction.
   */
  template <int spacedim, typename Number>
  std::vector<BoundingBox<spacedim, Number>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
                &patches,
    const double extra_ghost_cell_fraction)
  {
    Assert(
      extra_ghost_cell_fraction >= 0.0,
      ExcMessage(
        "The fraction of additional ghost cells to add must be positive."));
    // Set up patch bounding boxes and put patches in patches_to_elements:
    std::vector<BoundingBox<spacedim, Number>> patch_bboxes;
    for (const auto &patch_p : patches)
      {
        const SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM>>
                            pgeom = patch_p->getPatchGeometry();
        const double *const dx    = pgeom->getDx();

        BoundingBox<spacedim, Number> bbox;
        for (unsigned int d = 0; d < spacedim; ++d)
          {
            bbox.get_boundary_points().first[d] =
              pgeom->getXLower()[d] - extra_ghost_cell_fraction * dx[d];
            bbox.get_boundary_points().second[d] =
              pgeom->getXUpper()[d] + extra_ghost_cell_fraction * dx[d];
          }
        patch_bboxes.emplace_back(bbox);
      }

    return patch_bboxes;
  }

  template <int spacedim>
  std::vector<std::vector<hier::Box<spacedim>>>
  compute_nonoverlapping_patch_boxes(
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> &c_level,
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> &f_level)
  {
    const tbox::Pointer<hier::PatchLevel<spacedim>> coarse_level = c_level;
    AssertThrow(coarse_level, ExcFDLNotImplemented());
    const tbox::Pointer<hier::PatchLevel<spacedim>> fine_level = f_level;
    AssertThrow(fine_level, ExcFDLNotImplemented());
    AssertThrow(coarse_level->getLevelNumber() + 1 ==
                  fine_level->getLevelNumber(),
                ExcFDLNotImplemented());
    const hier::IntVector<spacedim> ratio =
      fine_level->getRatioToCoarserLevel();

    // Get all (including those not on this processor) fine-level boxes:
    hier::BoxList<spacedim> finer_box_list;
    long                    combined_size = 0;
    for (int i = 0; i < fine_level->getNumberOfPatches(); ++i)
      {
        hier::Box<spacedim> patch_box = fine_level->getBoxForPatch(i);
        patch_box.coarsen(ratio);
        combined_size += patch_box.size();
        finer_box_list.addItem(patch_box);
      }
    finer_box_list.simplifyBoxes();

    // Remove said boxes from each coarse-level patch:
    std::vector<std::vector<hier::Box<spacedim>>> result;
    long                                          coarse_size = 0;
    for (int i = 0; i < coarse_level->getNumberOfPatches(); ++i)
      {
        hier::BoxList<spacedim> coarse_box_list;
        coarse_box_list.addItem(coarse_level->getBoxForPatch(i));
        coarse_size += coarse_box_list.getFirstItem().size();
        coarse_box_list.removeIntersections(finer_box_list);

        result.emplace_back();
        std::vector<hier::Box<spacedim>> &boxes = result.back();
        typename tbox::List<hier::Box<spacedim>>::Iterator it(coarse_box_list);
        while (it)
          {
            boxes.push_back(*it);
            combined_size += boxes.back().size();
            it++;
          }
      }

    AssertThrow(coarse_size == combined_size, ExcFDLInternalError());

    return result;
  }

  template <int dim, int spacedim, typename Number>
  std::vector<BoundingBox<spacedim, Number>>
  compute_cell_bboxes(const DoFHandler<dim, spacedim> &dof_handler,
                      const Mapping<dim, spacedim>    &mapping)
  {
    // TODO: support multiple FEs
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();
    // TODO: also check bboxes by position of quadrature points instead of
    // just nodes. Use QProjector to place points solely on cell boundaries.
    const Quadrature<dim> nodal_quad(fe.get_unit_support_points());

    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      nodal_quad,
                                      update_quadrature_points);

    std::vector<BoundingBox<spacedim, Number>> bboxes;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          const BoundingBox<spacedim> dbox(fe_values.get_quadrature_points());
          // we have to do a conversion if Number != double
          BoundingBox<spacedim, Number> fbox;
          fbox.get_boundary_points() = dbox.get_boundary_points();
          bboxes.push_back(fbox);
        }
    return bboxes;
  }

  template <int dim, int spacedim, typename Number>
  std::vector<BoundingBox<spacedim, Number>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<dim, spacedim> &tria,
    const std::vector<BoundingBox<spacedim, Number>> &local_active_cell_bboxes)
  {
    Assert(
      tria.n_locally_owned_active_cells() == local_active_cell_bboxes.size(),
      ExcMessage("There should be a local bbox for each local active cell"));

    MPI_Datatype   mpi_type  = {};
    constexpr bool is_float  = std::is_same<Number, float>::value;
    constexpr bool is_double = std::is_same<Number, double>::value;
    static_assert(is_float || is_double, "Must be float or double");
    if (is_float)
      mpi_type = MPI_FLOAT;
    else if (is_double)
      mpi_type = MPI_DOUBLE;
    else
      AssertThrow(false, ExcFDLNotImplemented());

    constexpr auto n_nums_per_bbox = spacedim * 2;
    static_assert(sizeof(BoundingBox<spacedim, Number>) ==
                    sizeof(Number) * n_nums_per_bbox,
                  "packing failed");

    std::vector<BoundingBox<spacedim, Number>> global_bboxes(
      tria.n_active_cells());

    MPI_Comm comm = tria.get_communicator();
    // Exchange number of cells:
    const int        n_procs = Utilities::MPI::n_mpi_processes(comm);
    std::vector<int> bbox_entries_per_proc(n_procs);
    const int        bbox_entries_on_this_proc =
      tria.n_locally_owned_active_cells() * n_nums_per_bbox;
    int ierr = MPI_Allgather(&bbox_entries_on_this_proc,
                             1,
                             MPI_INT,
                             &bbox_entries_per_proc[0],
                             1,
                             MPI_INT,
                             comm);
    AssertThrowMPI(ierr);
    Assert(std::accumulate(bbox_entries_per_proc.begin(),
                           bbox_entries_per_proc.end(),
                           0u) == (tria.n_active_cells() * n_nums_per_bbox),
           ExcMessage("Should be a partition"));

    // Determine indices into temporary array:
    std::vector<int> offsets(n_procs);
    offsets[0] = 0;
    std::partial_sum(bbox_entries_per_proc.begin(),
                     bbox_entries_per_proc.end() - 1,
                     offsets.begin() + 1);
    // Communicate bboxes:
    std::vector<BoundingBox<spacedim, Number>> temp_bboxes(
      tria.n_active_cells());
    ierr = MPI_Allgatherv(reinterpret_cast<const Number *>(
                            local_active_cell_bboxes.data()),
                          bbox_entries_on_this_proc,
                          mpi_type,
                          temp_bboxes.data(),
                          bbox_entries_per_proc.data(),
                          offsets.data(),
                          mpi_type,
                          comm);
    AssertThrowMPI(ierr);

    // Copy to the correct ordering. Keep track of how many cells we have copied
    // from each processor:
    std::vector<int> current_proc_cell_n(n_procs);
    for (const auto &cell : tria.active_cell_iterators())
      {
        const types::subdomain_id this_cell_proc_n =
          tria.get_true_subdomain_ids_of_cells()[cell->active_cell_index()];
        global_bboxes[cell->active_cell_index()] =
          temp_bboxes[offsets[this_cell_proc_n] / n_nums_per_bbox +
                      current_proc_cell_n[this_cell_proc_n]];
        ++current_proc_cell_n[this_cell_proc_n];
      }

#ifdef DEBUG
    for (const auto &bbox : global_bboxes)
      Assert(bbox.volume() > 0, ExcMessage("bboxes should not be empty"));
#endif
    return global_bboxes;
  }

  template <int spacedim>
  BoundingBox<spacedim>
  box_to_bbox(
    const hier::Box<spacedim>                           &box,
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> &base_patch_level)
  {
    const tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
      base_patch_level;
    AssertThrow(patch_level, ExcFDLNotImplemented());
    const tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geom =
      patch_level->getGridGeometry();
    AssertThrow(grid_geom, ExcFDLNotImplemented());

    std::pair<Point<spacedim>, Point<spacedim>> result;
    const auto                                  ratio = patch_level->getRatio();
    for (unsigned int d = 0; d < spacedim; ++d)
      {
        result.first[d] = grid_geom->getXLower()[d] +
                          box.lower()(d) * grid_geom->getDx()[d] / ratio(d);
        result.second[d] = grid_geom->getXLower()[d] + (box.upper()(d) + 1) *
                                                         grid_geom->getDx()[d] /
                                                         ratio(d);
      }

    return BoundingBox<spacedim>(result);
  }

  // these depend on SAMRAI types, and SAMRAI only has 2D and 3D libraries, so
  // use whatever IBTK is using

  // compute_patch_bboxes:
  template std::vector<BoundingBox<NDIM, float>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
                &patches,
    const double extra_ghost_cell_fraction);

  template std::vector<BoundingBox<NDIM, double>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
                &patches,
    const double extra_ghost_cell_fraction);

  // compute_nonoverlapping_patch_boxes:
  template std::vector<std::vector<hier::Box<NDIM>>>
  compute_nonoverlapping_patch_boxes(
    const tbox::Pointer<hier::BasePatchLevel<NDIM>> &c_level,
    const tbox::Pointer<hier::BasePatchLevel<NDIM>> &f_level);

  // compute_cell_bboxes:
  template std::vector<BoundingBox<NDIM, float>>
  compute_cell_bboxes(const DoFHandler<NDIM - 1, NDIM> &dof_handler,
                      const Mapping<NDIM - 1, NDIM>    &mapping);

  template std::vector<BoundingBox<NDIM, float>>
  compute_cell_bboxes(const DoFHandler<NDIM, NDIM> &dof_handler,
                      const Mapping<NDIM, NDIM>    &mapping);

  template std::vector<BoundingBox<NDIM, double>>
  compute_cell_bboxes(const DoFHandler<NDIM - 1, NDIM> &dof_handler,
                      const Mapping<NDIM - 1, NDIM>    &mapping);

  template std::vector<BoundingBox<NDIM, double>>
  compute_cell_bboxes(const DoFHandler<NDIM, NDIM> &dof_handler,
                      const Mapping<NDIM, NDIM>    &mapping);

  // collect_all_active_cell_bboxes:
  template std::vector<BoundingBox<NDIM, float>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<NDIM - 1, NDIM> &tria,
    const std::vector<BoundingBox<NDIM, float>> &local_active_cell_bboxes);

  template std::vector<BoundingBox<NDIM, float>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<NDIM, NDIM> &tria,
    const std::vector<BoundingBox<NDIM, float>> &local_active_cell_bboxes);

  template std::vector<BoundingBox<NDIM, double>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<NDIM - 1, NDIM> &tria,
    const std::vector<BoundingBox<NDIM, double>> &local_active_cell_bboxes);

  template std::vector<BoundingBox<NDIM, double>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<NDIM, NDIM> &tria,
    const std::vector<BoundingBox<NDIM, double>> &local_active_cell_bboxes);

  template BoundingBox<NDIM>
  box_to_bbox(const hier::Box<NDIM>                           &box,
              const tbox::Pointer<hier::BasePatchLevel<NDIM>> &patch_level);
} // namespace fdl
