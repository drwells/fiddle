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
    local_processor_min_cell_index =
      std::numeric_limits<types::global_cell_index>::max();
    unsigned int n_locally_owned_cells = 0;

    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          local_processor_min_cell_index =
            std::min(local_processor_min_cell_index,
                     cell->global_active_cell_index());
          ++n_locally_owned_cells;
        }

    for (const auto &fiber_vec : fibers)
      AssertThrow(n_locally_owned_cells == fiber_vec.size(),
                  ExcMessage("Not enough tensors in this vector"));

    const auto n_fiber_networks = fibers.size();
    this->fibers.reinit(n_locally_owned_cells, n_fiber_networks);
    for (unsigned int j = 0; j < n_fiber_networks; ++j)
      for (unsigned int i = 0; i < n_locally_owned_cells; ++i)
        this->fibers(i, j) = fibers[j][i];
  }

  template class FiberNetwork<NDIM - 1, NDIM>;
  template class FiberNetwork<NDIM, NDIM>;
} // namespace fdl
