#ifndef included_fiddle_intersection_predicate_h
#define included_fiddle_intersection_predicate_h

#include <fiddle/base/config.h>

#include <deal.II/grid/tria.h>

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
} // namespace fdl

#endif
