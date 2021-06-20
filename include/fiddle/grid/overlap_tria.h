#ifndef included_fiddle_overlap_tria_h
#define included_fiddle_overlap_tria_h

#include <fiddle/base/exceptions.h>

#include <fiddle/grid/intersection_predicate.h>

#include <deal.II/base/subscriptor.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/tria.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Class describing a Triangulation built from a shared Triangulation where
   * each processor contains the subset of the input triangulation that overlaps
   * with the provided set of bounding boxes. In general, these subsets will
   * overlap (i.e., a cell can be assigned to an arbitrary number of
   * processors).
   *
   * This class is inherently serial (i.e., a single processor) since the cells
   * it stores have no notion of ghost cells of cells belonging to off-processor
   * overlap triangulations. Hence the communicator it stores is still
   * <code>MPI_COMM_SELF</code>.
   */
  template <int dim, int spacedim = dim>
  class OverlapTriangulation : public dealii::Triangulation<dim, spacedim>
  {
    using active_cell_iterator =
      typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;
    using cell_iterator =
      typename dealii::Triangulation<dim, spacedim>::cell_iterator;

  public:
    OverlapTriangulation() = default;

    OverlapTriangulation(
      const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
      const IntersectionPredicate<dim, spacedim> &          predicate);

    virtual types::subdomain_id
    locally_owned_subdomain() const;

    void
    reinit(const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
           const IntersectionPredicate<dim, spacedim> &          predicate);

    const parallel::shared::Triangulation<dim, spacedim> &
    get_native_triangulation() const;

    /**
     * Get the native cell iterator equivalent to the current cell iterator.
     */
    cell_iterator
    get_native_cell(const cell_iterator &cell) const;

    /**
     * Get the CellId for the corresponding cell on the native Triangulation.
     */
    CellId
    get_native_cell_id(const cell_iterator &cell) const;

    /**
     * Get the rank of the corresponding cell on the native Triangulation.
     */
    types::subdomain_id
    get_native_cell_subdomain_id(const cell_iterator &cell) const;

  protected:
    /**
     * Utility function that stores a native cell and returns its array index
     * (which will then be set as the user index or material id).
     */
    std::size_t
    add_native_cell(const cell_iterator &cell);

    void
    reinit_overlapping_tria(
      const IntersectionPredicate<dim, spacedim> &predicate);

    /**
     * Pointer to the Triangulation which describes the whole domain.
     */
    SmartPointer<const parallel::shared::Triangulation<dim, spacedim>,
                 OverlapTriangulation<dim, spacedim>>
      native_tria;

    /**
     * Level and index pairs (i.e., enough to create an iterator) of native
     * cells which have an equivalent cell on this triangulation.
     */
    std::vector<std::pair<int, int>> native_cells;

    /**
     * CellIds for native cells.
     */
    std::vector<CellId> native_cell_ids;

    /**
     * Subdomain ids of native cells, indexed by the corresponding overlapping
     * cells' active cell index.
     */
    std::vector<types::subdomain_id> native_cell_subdomain_ids;

    /**
     * Active cell iterators sorted by the active cell index of the
     * corresponding native cell. Useful for doing data transfer.
     */
    std::vector<active_cell_iterator> cell_iterators_in_active_native_order;
  };


  //
  // inline functions
  //
  template <int dim, int spacedim>
  const parallel::shared::Triangulation<dim, spacedim> &
  OverlapTriangulation<dim, spacedim>::get_native_triangulation() const
  {
    return *native_tria;
  }



  template <int dim, int spacedim>
  inline typename OverlapTriangulation<dim, spacedim>::cell_iterator
  OverlapTriangulation<dim, spacedim>::get_native_cell(
    const cell_iterator &cell) const
  {
    AssertIndexRange(cell->user_index(), native_cells.size());
    const auto          pair = native_cells[cell->user_index()];
    const cell_iterator native_cell(native_tria, pair.first, pair.second);
    Assert((native_cell->barycenter() - cell->barycenter()).norm() < 1e-12,
           ExcFDLInternalError());
    return native_cell;
  }



  template <int dim, int spacedim>
  inline CellId
  OverlapTriangulation<dim, spacedim>::get_native_cell_id(
    const cell_iterator &cell) const
  {
    AssertIndexRange(cell->user_index(), native_cells.size());
    return native_cell_ids[cell->user_index()];
  }



  template <int dim, int spacedim>
  inline types::subdomain_id
  OverlapTriangulation<dim, spacedim>::get_native_cell_subdomain_id(
    const cell_iterator &cell) const
  {
    AssertIndexRange(cell->user_index(), native_cells.size());
    return native_cell_subdomain_ids[cell->user_index()];

  }



  template <int dim, int spacedim>
  inline std::size_t
  OverlapTriangulation<dim, spacedim>::add_native_cell(
    const cell_iterator &cell)
  {
    Assert(&cell->get_triangulation() == native_tria,
           ExcMessage("should be a native cell"));
    native_cells.emplace_back(cell->level(), cell->index());
    native_cell_ids.emplace_back(cell->id());
    // During construction its useful to add nonactive cells, which aren't owned
    // by any process - permit that here too
    if (cell->is_active())
      native_cell_subdomain_ids.emplace_back(cell->subdomain_id());
    else
      native_cell_subdomain_ids.emplace_back(numbers::invalid_subdomain_id);
    return native_cells.size() - 1;
  }
} // namespace fdl

#endif // define included_fiddle_overlap_tria_h
