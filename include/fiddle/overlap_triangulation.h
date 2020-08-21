#ifndef included_fiddle_overlap_triangulation_h
#define included_fiddle_overlap_triangulation_h

#include <deal.II/base/bounding_box.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/distributed/shared_tria.h>

#include <algorithm>
#include <vector>

namespace fdl
{
using namespace dealii;

// todo - add it to deal.II
template <int spacedim>
bool intersects(const BoundingBox<spacedim> &a,
                const BoundingBox<spacedim> &b)
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
        if (!((a.lower_bound(d) <= b.lower_bound(d) && b.lower_bound(d) <= a.upper_bound(d)) ||
              (a.lower_bound(d) <= b.upper_bound(d) && b.upper_bound(d) <= a.upper_bound(d))) &&
            !((b.lower_bound(d) <= a.lower_bound(d) && a.lower_bound(d) <= b.upper_bound(d))))
        {
            return false;
        }
    }

    return true;
}



//
// Class describing a Triangulation built from a shared Triangulation.
//
template <int dim, int spacedim = dim>
class OverlapTriangulation : public Triangulation<dim, spacedim>
{
  using active_cell_iterator =
    typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;
  using cell_iterator =
    typename dealii::Triangulation<dim, spacedim>::cell_iterator;

  public:
    OverlapTriangulation
    (const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
     const std::vector<BoundingBox<spacedim>> &bboxes)
    {
      reinit(shared_tria, bboxes);
    }

    void
    reinit(const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
           const std::vector<BoundingBox<spacedim>> &bboxes)
    {
      // todo - clear signals, etc. if there is a new shared tria
      native_tria = &shared_tria;
      cell_iterators_in_active_native_order.clear();

      reinit_overlapping_tria(bboxes);

      // Also set up some cached information:
      for (const auto &cell : this->active_cell_iterators())
        cell_iterators_in_active_native_order.push_back(cell);
      std::sort(cell_iterators_in_active_native_order.begin(),
                cell_iterators_in_active_native_order.end(),
                [&](const auto &a, const auto &b)
                {return this->get_native_cell(a)->active_cell_index() <
                        this->get_native_cell(b)->active_cell_index();});
    }

    parallel::shared::Triangulation<dim> &get_native_triangulation()
    {
      return *native_tria;
    }

    /**
     * Get the native cell iterator equivalent to the current cell iterator.
     */
    inline
    active_cell_iterator
    get_native_cell(const active_cell_iterator &cell) const
    {
      return intersecting_native_cells[cell->user_index()];
    }

    inline
    const std::vector<active_cell_iterator> &
    get_cell_iterators_in_active_native_order() const
    {
      return cell_iterators_in_active_native_order;
    }

  protected:
    void
    reinit_overlapping_tria(const std::vector<BoundingBox<spacedim>> &bboxes)
    {
      intersecting_native_cells.clear();
      this->clear();

      std::vector<CellData<dim>> cells;
      SubCellData subcell_data;
      for (const auto &cell : native_tria->active_cell_iterators())
        {
          // TODO: use an rtree instead of checking boxes in a loop
          for (const auto &bbox : bboxes)
            if (intersects(cell->bounding_box(), bbox))
              {
                CellData<spacedim> cell_data;
                // Temporarily refer to native cells with the material id
                intersecting_native_cells.push_back(cell);
                cell_data.material_id = intersecting_native_cells.size() - 1;

                cell_data.vertices.clear();
                for (const auto &index : cell->vertex_indices())
                  cell_data.vertices.push_back(cell->vertex_index(index));

                cells.push_back(std::move(cell_data));
                // TODO: also populate subcell data
                // TODO: get multiple active levels working
              }
        }
      this->create_triangulation(native_tria->get_vertices(), cells, subcell_data);

      for (auto &cell : this->active_cell_iterators())
        {
          const auto &native_cell = intersecting_native_cells[cell->user_index()];
          cell->set_user_index(cell->material_id());
          cell->set_material_id(native_cell->material_id());
          cell->set_subdomain_id(native_cell->subdomain_id());
          cell->set_manifold_id(native_cell->manifold_id());
        }
    }

    /**
     * Pointer to the Triangulation which describes the whole domain.
     */
    SmartPointer<const parallel::shared::Triangulation<dim, spacedim>,
                 OverlapTriangulation<dim, spacedim>> native_tria;

    /**
     * Iterators to native cells which have an equivalent cell on this
     * triangulation.
     */
    std::vector<active_cell_iterator> intersecting_native_cells;

    /**
     * Active cell iterators sorted by the active cell index of the
     * corresponding native cell. Useful for doing data transfer.
     */
    std::vector<active_cell_iterator> cell_iterators_in_active_native_order;
};
}

#endif // define included_fiddle_overlap_triangulation_h
