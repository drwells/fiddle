#ifndef included_fiddle_grid_box_utilities_h
#define included_fiddle_grid_box_utilities_h

#include <deal.II/base/bounding_box.h>

#include <CartesianPatchGeometry.h>
#include <Patch.h>

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

  /**
   * Extract the bounding boxes for a set of SAMRAI patches. In addition, if
   * necessary, expand each bounding box by @p extra_ghost_cell_fraction times
   * the length of a cell in each coordinate direction.
   */
  template <int spacedim>
  std::vector<BoundingBox<spacedim>>
  compute_patch_bboxes(const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>> &patches,
                       const double extra_ghost_cell_fraction = 0.0)
  {
    Assert(extra_ghost_cell_fraction >= 0.0,
           ExcMessage("The fraction of additional ghost cells to add must be positive."));
    // Set up patch bounding boxes and put patches in patches_to_elements:
    std::vector<BoundingBox<spacedim>> patch_bboxes;
    for (const auto &patch_p : patches)
    {
      const SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianPatchGeometry<NDIM> > pgeom =
        patch_p->getPatchGeometry();
      const double* const dx = pgeom->getDx();

      BoundingBox<spacedim> bbox;
      for (unsigned int d = 0; d < spacedim; ++d)
      {
        bbox.get_boundary_points().first[d] = pgeom->getXLower()[d] - extra_ghost_cell_fraction*dx[d];
        bbox.get_boundary_points().second[d] = pgeom->getXUpper()[d] + extra_ghost_cell_fraction*dx[d];
      }
      patch_bboxes.emplace_back(bbox);
    }

    return patch_bboxes;
  }

} // namespace fdl

#endif
