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
    this->patches = patches;
    Assert(cell_bboxes.size() == tria.n_active_cells(),
           ExcMessage("each active cell should have a bounding box."));

    const std::vector<BoundingBox<spacedim, Number>> patch_bboxes =
      compute_patch_bboxes<spacedim, Number>(patches,
                                             extra_ghost_cell_fraction);
    cells.resize(patches.size());

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
            cells[patch_n].push_back(cell);
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
