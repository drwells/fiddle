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

        const std::vector<BoundingBox<spacedim>> patch_bboxes =
          compute_patch_bboxes(patches, extra_ghost_cell_fraction);
        cells.resize(patches.size());

        // Speed up intersection by putting the patch bboxes in an rtree
        auto rtree = pack_rtree_of_indices(patch_bboxes);

        for (const auto cell : tria.active_cell_iterators())
          // TODO - if all cells are active or artificial, then we don't need this check
          if (cell->is_locally_owned())
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

  template class PatchMap<2, 2>;

  template class PatchMap<2, 3>;

  template class PatchMap<3, 3>;
}
