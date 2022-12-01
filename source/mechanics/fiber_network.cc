#include <fiddle/mechanics/fiber_network.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/tria.h>

namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim>
  FiberNetwork<dim, spacedim>::FiberNetwork(
    const Triangulation<dim, spacedim>                  &tria,
    const std::vector<std::vector<Tensor<1, spacedim>>> &fibers)
    : tria(&tria)
  {
    // initialize local_processor_min_cell_index to maximum value it could be
    // set to
    local_processor_min_cell_index =
      std::numeric_limits<types::global_cell_index>::max();

    for (const auto &cell : tria.active_cell_iterators())
      {
        local_processor_min_cell_index =
          std::min(local_processor_min_cell_index, cell->active_cell_index());
      }

    // number of active cells in current processor
    auto n_table_rows = tria.n_active_cells();

    if (fibers[0].size() > 0)
      {
        AssertThrow(
          n_table_rows == fibers[0].size(),
          ExcMessage(
            "fibers vector size should be equal to the number of active cells"));
      }

    const auto n_table_cols = fibers.size();
    this->fibers.reinit(n_table_rows, n_table_cols);
    for (unsigned int i = 0; i < n_table_rows; ++i)
      for (unsigned int j = 0; j < n_table_cols; ++j)
        this->fibers(i, j) = fibers[j][i];
  }

  template <int dim, int spacedim>
  ArrayView<const Tensor<1, spacedim>>
  FiberNetwork<dim, spacedim>::get_fibers(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    // calculate the correct vector entry
    auto cell_index =
      cell->global_active_cell_index() - local_processor_min_cell_index;

    return make_array_view(fibers, cell_index, 0, fibers.size(1));
  }

  template class FiberNetwork<NDIM - 1, NDIM>;
  template class FiberNetwork<NDIM, NDIM>;
} // namespace fdl
