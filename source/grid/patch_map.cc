#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/patch_map.h>

#include <deal.II/numerics/rtree.h>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim>
  template <typename Number>
  PatchMap<dim, spacedim>::PatchMap(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const double                                      extra_ghost_cell_fraction,
    const Triangulation<dim, spacedim> &              tria,
    const std::vector<BoundingBox<spacedim, Number>> &cell_bboxes)
  {
    reinit(patches, extra_ghost_cell_fraction, tria, cell_bboxes);
  }

  template <int dim, int spacedim>
  template <typename Number>
  void
  PatchMap<dim, spacedim>::reinit(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const double                                      extra_ghost_cell_fraction,
    const Triangulation<dim, spacedim> &              tria,
    const std::vector<BoundingBox<spacedim, Number>> &cell_bboxes)
  {
    this->tria    = &tria;
    this->patches = patches;
    Assert(cell_bboxes.size() == tria.n_active_cells(),
           ExcMessage("each active cell should have a bounding box."));

    const std::vector<BoundingBox<spacedim, Number>> patch_bboxes =
      compute_patch_bboxes<spacedim, Number>(patches,
                                             extra_ghost_cell_fraction);
    patch_level_cells.clear();
    patch_level_cells.resize(patches.size());
    for (unsigned int patch_n = 0; patch_n < patches.size(); ++patch_n)
      {
        patch_level_cells[patch_n].resize(tria.n_levels());
        for (unsigned int level_n = 0; level_n < tria.n_levels(); ++level_n)
          patch_level_cells[patch_n][level_n].set_size(tria.n_cells(level_n));
      }

    // Speed up intersection by putting the patch bboxes in an rtree
    const auto rtree = pack_rtree_of_indices(patch_bboxes);
    for (const auto &cell : tria.active_cell_iterators())
      {
        const BoundingBox<spacedim, Number> &cell_bbox =
          cell_bboxes[cell->active_cell_index()];

        namespace bgi = boost::geometry::index;
        for (const std::size_t patch_n :
             rtree | bgi::adaptors::queried(bgi::intersects(cell_bbox)))
          {
            AssertIndexRange(patch_n, patches.size());
            AssertIndexRange(cell->level(), patch_level_cells[patch_n].size());
            patch_level_cells[patch_n][cell->level()].add_index(cell->index());
          }
      }

    for (auto &level_cells : patch_level_cells)
      for (auto &cell_indices : level_cells)
        cell_indices.compress();

    // update cummulative number of cells
    cummulative_n_cells.clear();
    cummulative_n_cells.resize(patches.size());
    for (unsigned int patch_n = 0; patch_n < patches.size(); ++patch_n)
      {
        for (unsigned int level_n = 0; level_n < tria.n_levels(); ++level_n)
          {
            cummulative_n_cells[patch_n].push_back(
              patch_level_cells[patch_n][level_n].n_elements());
            if (level_n != 0)
              cummulative_n_cells[patch_n].back() +=
                cummulative_n_cells[patch_n][level_n - 1];
          }
      }
  }

  // Since we depend on SAMRAI types (and SAMRAI uses 2D or 3D libraries) we
  // instantiate based on NDIM (provided by IBTK)

  template class PatchMap<NDIM - 1, NDIM>;

  template PatchMap<NDIM - 1, NDIM>::PatchMap(
    const std::vector<tbox::Pointer<hier::Patch<NDIM>>> &,
    const double,
    const Triangulation<NDIM - 1, NDIM> &,
    const std::vector<BoundingBox<NDIM, float>> &cell_bboxes);

  template PatchMap<NDIM - 1, NDIM>::PatchMap(
    const std::vector<tbox::Pointer<hier::Patch<NDIM>>> &,
    const double,
    const Triangulation<NDIM - 1, NDIM> &,
    const std::vector<BoundingBox<NDIM, double>> &cell_bboxes);

  template class PatchMap<NDIM, NDIM>;

  template PatchMap<NDIM, NDIM>::PatchMap(
    const std::vector<tbox::Pointer<hier::Patch<NDIM>>> &,
    const double,
    const Triangulation<NDIM, NDIM> &,
    const std::vector<BoundingBox<NDIM, float>> &cell_bboxes);

  template PatchMap<NDIM, NDIM>::PatchMap(
    const std::vector<tbox::Pointer<hier::Patch<NDIM>>> &,
    const double,
    const Triangulation<NDIM, NDIM> &,
    const std::vector<BoundingBox<NDIM, double>> &cell_bboxes);

} // namespace fdl
