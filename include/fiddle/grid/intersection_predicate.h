#ifndef included_fiddle_intersection_predicate_h
#define included_fiddle_intersection_predicate_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>

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
    TriaIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &bboxes)
      : patch_boxes(bboxes)
    {}

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      const auto cell_bbox = cell->bounding_box();
      for (const auto &bbox : patch_boxes)
        if (intersects(cell_bbox, bbox))
          return true;
      return false;
    }

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
      const parallel::shared::Triangulation<dim, spacedim> &tria)
      : tria(&tria)
      , active_cell_bboxes(a_cell_bboxes)
      , patch_bboxes(p_bboxes)
    {}

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      Assert(&cell->get_triangulation() == tria,
             ExcMessage("only valid for inputs constructed from the originally "
                        "provided Triangulation"));
      // If the cell is active check its bbox:
      if (cell->is_active())
        {
          const auto &cell_bbox = active_cell_bboxes[cell->active_cell_index()];
          for (const auto &bbox : patch_bboxes)
            if (intersects(cell_bbox, bbox))
              return true;
          return false;
        }
      // Otherwise see if it has a descendant that intersects:
      else if (cell->has_children())
        {
          const auto n_children             = cell->n_children();
          bool       has_intersecting_child = false;
          for (unsigned int child_n = 0; child_n < n_children; ++child_n)
            {
              const bool child_intersects = (*this)(cell->child(child_n));
              if (child_intersects)
                {
                  has_intersecting_child = true;
                  break;
                }
            }
          return has_intersecting_child;
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }

      Assert(false, ExcFDLInternalError());
      return false;
    }

    const SmartPointer<const Triangulation<dim, spacedim>> tria;
    const std::vector<BoundingBox<spacedim, float>>        active_cell_bboxes;
    const std::vector<BoundingBox<spacedim>>               patch_bboxes;
  };
} // namespace fdl

#endif
