#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/surface_meter.h>

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <tbox/Database.h>
#include <tbox/Pointer.h>

#include <cmath>
#include <fstream>

#include "../tests.h"

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);

  const unsigned int           n_points = 8;
  std::vector<Point<spacedim>> boundary_points(n_points);
  for (unsigned int n = 0; n < n_points; ++n)
    {
      boundary_points[n][0] =
        std::cos(2.0 * numbers::PI * n / double(n_points));
      boundary_points[n][1] =
        std::sin(2.0 * numbers::PI * n / double(n_points));
      boundary_points[n][2] =
        std::sin(4.0 * numbers::PI * n / double(n_points));
    }

  std::vector<Tensor<1, spacedim>> velocities(n_points);

  fdl::SurfaceMeter<dim, spacedim> test_meter(boundary_points,
                                              velocities,
                                              patch_hierarchy);

  // const Triangulation<dim - 1, spacedim> &test_tria =
  // test_meter.get_triangulation();
  const std::vector<Point<spacedim>> &vertex_points =
    test_meter.get_triangulation().get_vertices();

  std::ofstream output("output");
  for (unsigned int i = 0; i < n_points; ++i)
    {
      output << "original boundary point: " << boundary_points[i] << std::endl;
      output << "generated boundary point: " << vertex_points[i] << std::endl
             << std::endl;
    }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit                      ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv);

  test<3>(app_initializer);
}
