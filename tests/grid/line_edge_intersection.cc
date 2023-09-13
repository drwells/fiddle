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

      Triangulation<2> tria_bulk;
      Triangulation<1,2> tria;
      GridGenerator::hyper_ball(tria_bulk);
      GridGenerator::hyper_sphere(tria,Point< 2 >(0,0), 1 );
      tria.refine_global(2);

      std::vector<BoundingBox<2>> cell_bboxes;
      for (const auto &cell : tria.active_cell_iterators())
        cell_bboxes.push_back(cell->bounding_box());
      // Set up the relevant fiddle class:
      dealii::Point<2> r(0.9999999,0.0);
      dealii::Tensor<1,2> q;
      q[0]=0;
      q[1]=1;
      const MappingQ<1, 2> mapping(1);

      std::vector<std::pair<double, Point<1>> > t_vals;
      // now do the actual test
      FE_Nothing<1,2> fe;
      DoFHandler<1,2> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
            std::array<Point<2>, 2> Pts ={mapping.transform_unit_to_real_cell(cell, Point<1>(0)),mapping.transform_unit_to_real_cell(cell, Point<1>(1))};
            fdl::intersect_line_with_element<1,2>(t_vals,Pts,r,q,-0.000);
        }
}