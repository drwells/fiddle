#include <fiddle/grid/patch_map.h>

#include <deal.II/numerics/rtree.h>

namespace fdl
{
    using namespace dealii;

    template <int dim, int spacedim>
    PatchMap<dim, spacedim>::PatchMap(
      const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>> &patches,
      const double extra_ghost_cell_fraction,
      const Triangulation<dim, spacedim> &tria,
      std::vector<BoundingBox<spacedim>> &cell_bboxes)
      : patches(patches)
      {
        Assert(cell_bboxes.size() == tria.n_active_cells(),
               ExcMessage("each active cell should have a bounding box."));

        const std::vector<BoundingBox<spacedim, float>> patch_bboxes =
          compute_patch_bboxes<spacedim, float>(patches, extra_ghost_cell_fraction);
        cells.resize(patches.size());

        // Speed up intersection by putting the patch bboxes in an rtree
        const auto rtree = pack_rtree_of_indices(patch_bboxes);
        for (const auto &cell : tria.active_cell_iterators())
          {
            const BoundingBox<spacedim> &cell_bbox = cell_bboxes[cell->active_cell_index()];

            namespace bgi = boost::geometry::index;
            for (const std::size_t patch_n : rtree | bgi::adaptors::queried(bgi::intersects(cell_bbox)))
            {
              AssertIndexRange(patch_n, patches.size());
              cells[patch_n].push_back(cell);
            }
          }
      }

  // Since we depend on SAMRAI types (and SAMRAI uses 2D or 3D libraries) we
  // instantiate based on NDIM (provided by IBTK)

  template class PatchMap<NDIM - 1, NDIM>;
  template class PatchMap<NDIM, NDIM>;
}
