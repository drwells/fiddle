#ifndef included_fiddle_grid_box_utilities_h
#define included_fiddle_grid_box_utilities_h

#include <fiddle/grid/box_utilities.h>

#include <deal.II/base/bounding_box.h>

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
  template <int spacedim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
      &          patches,
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

  /**
   * Another helper function since MultiblockPatchLevel and PatchLevel have
   * slightly different interfaces
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<spacedim>>
                    base_patch_level)
  {
    using namespace SAMRAI;
    tbox::Pointer<hier::PatchLevel<spacedim>> patch_level(base_patch_level);
    tbox::Pointer<hier::MultiblockPatchLevel<spacedim>> multi_patch_level(
      base_patch_level);

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> result;
    auto get_patches = [&](tbox::Pointer<hier::PatchLevel<spacedim>> level) {
      for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
        result.emplace_back(level->getPatch(p()));
    };

    if (patch_level)
      {
        get_patches(patch_level);
      }
    else if (multi_patch_level)
      {
        for (int block_n = 0; block_n < multi_patch_level->getNumberOfBlocks();
             ++block_n)
          {
            get_patches(multi_patch_level->getPatchLevelForBlock(block_n));
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    return result;
  }

  /**
   * Helper function for extracting locally owned patches from a patch level.
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
    // Downcast and move to the more general function
    SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<spacedim>>
      base_patch_level(patch_level);
    return extract_patches(base_patch_level);
  }

  // these depend on SAMRAI types, and SAMRAI only has 2D and 3D libraries, so
  // use whatever IBTK is using

  template std::vector<BoundingBox<NDIM, float>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
      &          patches,
    const double extra_ghost_cell_fraction);

  template std::vector<BoundingBox<NDIM, double>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
      &          patches,
    const double extra_ghost_cell_fraction);

  template std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
  extract_patches(
    SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<NDIM>> patch_level);

  template std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>>>
  extract_patches(
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);
} // namespace fdl

#endif
