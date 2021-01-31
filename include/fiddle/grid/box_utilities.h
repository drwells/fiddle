#ifndef included_fiddle_grid_box_utilities_h
#define included_fiddle_grid_box_utilities_h

#include <deal.II/base/bounding_box.h>

#include <Patch.h>
#include <BasePatchLevel.h>
#include <PatchLevel.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Fast intersection check between two bounding boxes.
   */
  template <int spacedim, typename Number1, typename Number2>
  bool
  intersects(const BoundingBox<spacedim, Number1> &a,
             const BoundingBox<spacedim, Number2> &b);

  /**
   * Extract the bounding boxes for a set of SAMRAI patches. In addition, if
   * necessary, expand each bounding box by @p extra_ghost_cell_fraction times
   * the length of a cell in each coordinate direction.
   */
  template <int spacedim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>> &patches,
    const double extra_ghost_cell_fraction = 0.0);

  /**
   * Helper function for extracting locally owned patches from a base patch level.
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<spacedim>> base_patch_level);

  /**
   * Helper function for extracting locally owned patches from a patch level.
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level);


  // --------------------------- inline functions --------------------------- //


  template <int spacedim, typename Number1, typename Number2>
  inline
  bool
  intersects(const BoundingBox<spacedim, Number1> &a,
             const BoundingBox<spacedim, Number2> &b)
  {
    // Since boxes are tensor products of line intervals it suffices to check
    // that the line segments for each coordinate axis overlap.
    for (unsigned int d = 0; d < spacedim; ++d)
      {
        // Line segments can intersect in two ways:
        // 1. They can overlap.
        // 2. One can be inside the other.
        //
        // In the first case we want to see if either end point of the second
        // line segment lies within the first. In the second case we can simply
        // check that one end point of the first line segment lies in the second
        // line segment. Note that we don't need, in the second case, to do two
        // checks since that case is already covered by the first.
        if (!((a.lower_bound(d) <= b.lower_bound(d) &&
               b.lower_bound(d) <= a.upper_bound(d)) ||
              (a.lower_bound(d) <= b.upper_bound(d) &&
               b.upper_bound(d) <= a.upper_bound(d))) &&
            !((b.lower_bound(d) <= a.lower_bound(d) &&
               a.lower_bound(d) <= b.upper_bound(d))))
          {
            return false;
          }
      }
    return true;
  }
} // namespace fdl

#endif
