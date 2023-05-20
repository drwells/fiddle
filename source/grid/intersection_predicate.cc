#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/intersection_predicate.h>

namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim>
  TriaIntersectionPredicate<dim, spacedim>::TriaIntersectionPredicate(
    const std::vector<BoundingBox<spacedim>> &bboxes)
    : patch_boxes(bboxes)
  {}

  template <int dim, int spacedim>
  bool
  TriaIntersectionPredicate<dim, spacedim>::operator()(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell) const
  {
    const auto cell_bbox = cell->bounding_box();
    for (const auto &bbox : patch_boxes)
      if (intersects(cell_bbox, bbox))
        return true;
    return false;
  }

  template <int dim, int spacedim>
  BoxIntersectionPredicate<dim, spacedim>::BoxIntersectionPredicate(
    const std::vector<BoundingBox<spacedim, float>>      &a_cell_bboxes,
    const std::vector<BoundingBox<spacedim>>             &p_bboxes,
    const parallel::shared::Triangulation<dim, spacedim> &tria)
    : tria(&tria)
    , active_cell_bboxes(a_cell_bboxes)
    , patch_bboxes(p_bboxes)
  {}

  template <int dim, int spacedim>
  bool
  BoxIntersectionPredicate<dim, spacedim>::operator()(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell) const
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

  template class TriaIntersectionPredicate<1, 1>;
  template class TriaIntersectionPredicate<1, 2>;
  template class TriaIntersectionPredicate<1, 3>;
  template class TriaIntersectionPredicate<2, 2>;
  template class TriaIntersectionPredicate<2, 3>;
  template class TriaIntersectionPredicate<3, 3>;

  template class BoxIntersectionPredicate<1, 1>;
  template class BoxIntersectionPredicate<1, 2>;
  template class BoxIntersectionPredicate<1, 3>;
  template class BoxIntersectionPredicate<2, 2>;
  template class BoxIntersectionPredicate<2, 3>;
  template class BoxIntersectionPredicate<3, 3>;
} // namespace fdl
