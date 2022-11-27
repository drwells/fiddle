#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/table.h>

#include <fiddle/postprocess/fiber_network.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

#include "../tests.h"
#include "../../source/postprocess/fiber_network.cc"


using namespace SAMRAI;
using namespace dealii;

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
    const MPI_Comm mpi_comm = MPI_COMM_WORLD;
    auto           input_db = app_initializer->getInputDatabase();
    auto           test_db  = input_db->getDatabase("test");
    dealii::Triangulation<2,2> tria;

    // create 2D mesh with 4 elements    
    dealii::GridGenerator::hyper_cube(tria);
    tria.refine_global(1);

    // let's say all cells have constant fiber fields:
    dealii::Tensor<1, spacedim> f1, f2;
    f1[0] = 1;
    f1[1] = 0;
    f2[0] = 0;
    f2[1] = 1;

    std::vector<dealii::Tensor<1, spacedim>> fibers1;
    std::vector<dealii::Tensor<1, spacedim>> fibers2;

    for(unsigned int i=0; i<tria.n_active_cells(); i++)
    {
        fibers1.push_back(f1);
        fibers2.push_back(f2);
    }

    std::vector<std::vector<dealii::Tensor<1, spacedim>>> fibers;
    fibers.push_back(fibers1);
    fibers.push_back(fibers2);

    fdl::FiberNetwork<dim,spacedim> fiber_network(tria,fibers);

    std::ostringstream local_out;

    local_out << "Test with two constant vector fields\n";
   

   for (const auto cell_iterator : tria.active_cell_iterators())
    {
        auto cell = cell_iterator;
        auto array = fiber_network.get_fibers(cell);
        local_out << cell->active_cell_index() << " " << array[0] << " " << array[1] << std::endl;
    }

  std::ofstream output;
  if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    output.open("output");

  print_strings_on_0(local_out.str(), mpi_comm, output);
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit                      ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "fiber_network_01.log");

  test<NDIM>(app_initializer);
}
