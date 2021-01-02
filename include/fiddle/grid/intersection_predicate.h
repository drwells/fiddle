#ifndef included_fiddle_intersection_predicate_h
#define included_fiddle_intersection_predicate_h

#include <deal.II/base/bounding_box.h>

#include <deal.II/grid/tria.h>

namespace fdl
{
  using namespace dealii;

  // todo - add it to deal.II
  template <int spacedim>
  bool
  intersects(const BoundingBox<spacedim> &a, const BoundingBox<spacedim> &b)
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
   * Class which can determine whether or not a given cell intersects some
   * geometric object.
   */
  template <int dim, int spacedim = dim>
  class IntersectionPredicate
  {
  public:
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
    TriaIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &bboxes)
      : bounding_boxes(bboxes)
    {
      // TODO: build an rtree here.
    }

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      const auto cell_bbox = cell->bounding_box();
      for (const auto &bbox : bounding_boxes)
        if (intersects(cell_bbox, bbox))
          return true;
      return false;
    }

  protected:
    std::vector<BoundingBox<spacedim>> bounding_boxes;
  };

  /**
   * Intersection predicate based on a displacement from a finite element field.
   */
  template <int dim, int spacedim = dim>
  class FEIntersectionPredicate : public IntersectionPredicate<dim, spacedim>
  {
  public:
    FEIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &)
    {
      // TODO: implement.
    }

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      Assert(false, ExcNotImplemented());
      return false;
    }
  };
} // namespace fdl

#endif
