#ifndef included_fiddle_postprocess_fiber_network_h
#define included_fiddle_postprocess_fiber_network_h

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

namespace fdl
{
  using namespace dealii;
    
    template <int dim, int spacedim = dim>
    class FiberNetwork
    {
        /*
        stores several cell-centered fiber network data
        */
    public:

        FiberNetwork(const Triangulation<dim, spacedim> &tria,
                    const std::vector<std::vector<Tensor<1, spacedim>>> &fibers)
                    : tria(&tria)  
                    {
                        // initialize local_processor_min_cell_index to maximum value it could be set to
                        local_processor_min_cell_index = std::numeric_limits<types::global_cell_index>::max();

                        for (const auto &cell : tria.active_cell_iterators())
                        {
                            local_processor_min_cell_index = std::min(local_processor_min_cell_index, cell->active_cell_index());
                        }

                        //number of cells in the mesh
                        auto n_table_rows = tria.n_active_cells(); // = fibers_vec.size()

                        // number of fiber fields - should not cause segfault
                        auto n_table_cols = fibers.size();

                        this->fibers.reinit(n_table_rows,n_table_cols);

                        /* assumes fibers vector is such that the outermost layer 
                         has size of number of fiber networks
                         and that innermost one has the size of numbers of cells*/
                        for (unsigned int i = 0; i < n_table_rows; ++i)
                        {
                            for (unsigned int j = 0; j < n_table_cols; ++j)
                            {
                                this->fibers(i,j)=fibers[j][i];
                            }
                        }
                    };
        
        ~FiberNetwork(){};

        const ArrayView<const Tensor<1, spacedim>> &
        get_fibers(typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const;

    private:
        const SmartPointer<const Triangulation<dim, spacedim>> tria;
        Table<2, Tensor<1, dim>> fibers;
        types::global_cell_index local_processor_min_cell_index;
    };
}

#endif