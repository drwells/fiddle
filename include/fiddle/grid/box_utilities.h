#ifndef included_fiddle_grid_box_utilities_h
#define included_fiddle_grid_box_utilities_h

#include <fiddle/base/config.h>

FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/base/bounding_box.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <vector>

// forward declarations
namespace dealii
{
  template <int, int>
  class Mapping;
  template <int, int>
  class DoFHandler;

  namespace parallel
  {
    namespace shared
    {
      template <int, int>
      class Triangulation;
    }
  } // namespace parallel
} // namespace dealii

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class BasePatchLevel;
    template <int>
    class Box;
    template <int>
    class Patch;
  } // namespace hier

  namespace tbox
  {
    template <typename>
    class Pointer;
  }
} // namespace SAMRAI


namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Fast intersection check between two bounding boxes.
   */
  template <int spacedim, typename Number1, typename Number2>
  bool
  intersects(const BoundingBox<spacedim, Number1> &a,
             const BoundingBox<spacedim, Number2> &b);

  /**
   * Compute the bounding boxes for a set of SAMRAI patches. In addition, if
   * necessary, expand each bounding box by @p extra_ghost_cell_fraction times
   * the length of a cell in each coordinate direction.
   */
  template <int spacedim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_patch_bboxes(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const double extra_ghost_cell_fraction = 0.0);

  /**
   * For each patch in @p c_level, return the list of boxes which intersect that
   * patch but not any patch in @p f_level. This intersection may be empty. Like
   * the other functions this is only done for coarse patches owned by the
   * current processor.
   */
  template <int spacedim>
  std::vector<std::vector<hier::Box<spacedim>>>
  compute_nonoverlapping_patch_boxes(
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> &c_level,
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> &f_level);

  /**
   * Compute the bounding boxes for all locally owned and active cells for a
   * finite element field.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_cell_bboxes(const DoFHandler<dim, spacedim> &dof_handler,
                      const Mapping<dim, spacedim>    &mapping);

  /**
   * Collect all bounding boxes on all processors.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<dim, spacedim> &tria,
    const std::vector<BoundingBox<spacedim, Number>> &local_active_cell_bboxes);

  /**
   * Convert a Box (in SAMRAI's index space) to a BoundingBox (in real space).
   */
  template <int spacedim>
  BoundingBox<spacedim>
  box_to_bbox(const hier::Box<spacedim>                           &box,
              const tbox::Pointer<hier::BasePatchLevel<spacedim>> &patch_level);


  // --------------------------- inline functions --------------------------- //


  template <int spacedim, typename Number1, typename Number2>
  inline bool
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
