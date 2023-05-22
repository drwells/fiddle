#ifndef included_fiddle_intersection_predicate_h
#define included_fiddle_intersection_predicate_h

#include <fiddle/base/config.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/grid/tria.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Class which can determine whether or not a given cell intersects some
   * geometric object.
   *
   * At the present time, since fiddle only works with
   * parallel::shared::Triangulation, this class assumes that it can compute an
   * answer for <emph>any</emph> cell in the Triangulation, and not just locally
   * owned cells.
   */
  template <int dim, int spacedim = dim>
  class IntersectionPredicate
  {
  public:
    /**
     * See if a given cell intersects whatever geometric object this object
     * refers to.
     */
    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const = 0;

    virtual ~IntersectionPredicate() = default;
  };

  /**
   * Intersection predicate that determines intersections based on the locations
   * of cells in the Triangulation and nothing else.
   */
  template <int dim, int spacedim = dim>
  class TriaIntersectionPredicate : public IntersectionPredicate<dim, spacedim>
  {
  public:
    TriaIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &bboxes);

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override;

    const std::vector<BoundingBox<spacedim>> patch_boxes;
  };

  /**
   * Intersection predicate based on general bounding boxes for each cell.
   *
   * This class is intended for usage with parallel::shared::Triangulation. In
   * particular, the bounding boxes associated with all active cells will be
   * present on all processors. This is useful for creating an
   * OverlapTriangulation on each processor with bounding boxes intersecting an
   * arbitrary part of the Triangulation.
   */
  template <int dim, int spacedim = dim>
  class BoxIntersectionPredicate : public IntersectionPredicate<dim, spacedim>
  {
  public:
    BoxIntersectionPredicate(
      const std::vector<BoundingBox<spacedim, float>>      &a_cell_bboxes,
      const std::vector<BoundingBox<spacedim>>             &p_bboxes,
      const parallel::shared::Triangulation<dim, spacedim> &tria);

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override;

    const SmartPointer<const Triangulation<dim, spacedim>> tria;
    const std::vector<BoundingBox<spacedim, float>>        active_cell_bboxes;
    const std::vector<BoundingBox<spacedim>>               patch_bboxes;
  };
} // namespace fdl

#endif
