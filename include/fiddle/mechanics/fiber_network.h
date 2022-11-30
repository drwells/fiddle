#ifndef included_fiddle_postprocess_fiber_network_h
#define included_fiddle_postprocess_fiber_network_h

#include <deal.II/base/array_view.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/tria.h>



namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim = dim>
  class FiberNetwork
  {
    /*
     * Reads cell-centered fiber field(s) stored in a vector
     * and stores the data in a Table.
     */
  public:
    /*
     * Constructor
     */
    FiberNetwork(const Triangulation<dim, spacedim> &                 tria,
                 const std::vector<std::vector<Tensor<1, spacedim>>> &fibers);

    // function that returns an ArrayView of the fiber field(s) corresponding to
    // a cell
    ArrayView<const Tensor<1, spacedim>>
    get_fibers(
      typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const;

  private:
    const SmartPointer<const Triangulation<dim, spacedim>> tria;
    dealii::Table<2, dealii::Tensor<1, spacedim>>          fibers;
    types::global_cell_index local_processor_min_cell_index;
  };
} // namespace fdl

#endif