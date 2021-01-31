#include <fiddle/grid/box_utilities.h>

#include <deal.II/numerics/rtree.h>

#include <ibtk/IndexUtilities.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  template <int spacedim, typename Number, typename TYPE>
  void
  tag_cells_internal(const std::vector<BoundingBox<spacedim, Number>> &bboxes,
                     const int tag_index,
                     SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
    using namespace SAMRAI;

    const tbox::Pointer<geom::CartesianGridGeometry<spacedim>>
      grid_geom = patch_level->getGridGeometry();
    const hier::IntVector<spacedim> ratio = patch_level->getRatio();

    const std::vector<tbox::Pointer<hier::Patch<spacedim>>>
      patches = extract_patches(patch_level);

    std::vector<tbox::Pointer<pdat::CellData<spacedim, TYPE> > > tag_data;
    for (const auto &patch : patches)
    {
        Assert(patch->getPatch, ExcMessage("should be a pointer here"));
      tag_data.push_back(patch->getPatchData(tag_index));
    }
    const std::vector<BoundingBox<spacedim, float>>
      patch_bboxes = compute_patch_bboxes<spacedim, float>(patches);
    const auto rtree = pack_rtree_of_indices(patch_bboxes);

    // loop over element bboxes...
    for (const auto &bbox : bboxes)
    {
      const hier::Index<spacedim> i_lower
        = IBTK::IndexUtilities::getCellIndex(bbox.get_boundary_points().first,
                                             grid_geom, ratio);
      const hier::Index<spacedim> i_upper
        = IBTK::IndexUtilities::getCellIndex(bbox.get_boundary_points().second,
                                             grid_geom, ratio);
      const hier::Box<spacedim> box(i_lower, i_upper);

      // and determine which patches each intersects.
      namespace bgi = boost::geometry::index;
      // TODO: this still allocates memory. We should use something else to
      // avoid that, like boost::function_to_output_iterator or our own
      // equivalent
      for (const std::size_t patch_n : rtree |
             bgi::adaptors::queried(bgi::intersects(bbox)))
      {
        AssertIndexRange(patch_n, patches.size());
        tag_data[patch_n]->fillAll(TYPE(1), box);
      }
    }
  }

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(const std::vector<BoundingBox<spacedim, Number>> &bboxes,
            const int tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
    // SAMRAI doesn't offer a way to dispatch on data type so we have to do it ourselves
    using namespace SAMRAI;

    if (patch_level->getNumberOfPatches() == 0)
      {
        return;
      }
    else
      {
        for (typename hier::PatchLevel<spacedim>::Iterator p(patch_level); p; p++)
          {
            const tbox::Pointer<hier::Patch<spacedim>> patch = patch_level->getPatch(p());

            const tbox::Pointer<pdat::CellData<spacedim, int>> int_data
              = patch->getPatchData(tag_index);
            const tbox::Pointer<pdat::CellData<spacedim, float>> float_data
              = patch->getPatchData(tag_index);
            const tbox::Pointer<pdat::CellData<spacedim, double>> double_data
              = patch->getPatchData(tag_index);

            if (int_data)
              tag_cells_internal<spacedim, Number, int>(bboxes, tag_index, patch_level);
            else if (float_data)
              tag_cells_internal<spacedim, Number, float>(bboxes, tag_index, patch_level);
            else if (double_data)
              tag_cells_internal<spacedim, Number, double>(bboxes, tag_index, patch_level);

            break;
          }
      }
  }

    template
    void tag_cells(const std::vector<BoundingBox<NDIM, float>> &bboxes,
                   const int tag_index,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);

    template
    void tag_cells(const std::vector<BoundingBox<NDIM, double>> &bboxes,
                   const int tag_index,
                   SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);
} // namespace fdl
