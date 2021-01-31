#ifndef included_fiddle_interaction_interaction_h
#define included_fiddle_interaction_interaction_h

#include <deal.II/base/bounding_box.h>

#include <PatchLevel.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(const std::vector<BoundingBox<spacedim, Number>> &bboxes,
            const int tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level);
} // namespace fdl

#endif
