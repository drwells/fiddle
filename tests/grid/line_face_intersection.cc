#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/surface_meter.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/SAMRAIDataCache.h>

#include <fstream>

#include "../tests.h"
#include <fiddle/interaction/interaction_utilities.h>

using namespace dealii;
using namespace SAMRAI;

// Test the meter mesh code for a basic interpolation problem

int main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");
   Triangulation<2,3> tria;
   GridGenerator::hyper_sphere(tria,Point< 3 >(0,0,0), 1 );
   tria.refine_global(3);

   std::vector<BoundingBox<3>> cell_bboxes;
   for (const auto &cell : tria.active_cell_iterators())
     cell_bboxes.push_back(cell->bounding_box());
   // Set up the relevant fiddle class:
   dealii::Point<3> r(0.9999,0.000001, 0.000010);
   dealii::Tensor<1,3> q;
   q[0]=0;
   q[1]=1;
   q[2]=0;
   const MappingQ<2, 3> mapping(7);

   std::vector<std::pair<double, Point<2>> > t_vals;
   // now do the actual test
   FE_Nothing<2,3> fe;
   DoFHandler<2,3> dof_handler(tria);
   dof_handler.distribute_dofs(fe);
   for (const auto &cell : dof_handler.active_cell_iterators())
     {
       bool z=fdl::intersect_line_with_face(t_vals,cell,mapping,r,q,-0.000);
     }
return 1;
}
