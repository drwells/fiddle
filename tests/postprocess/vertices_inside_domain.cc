#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/surface_meter.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
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

using namespace dealii;
using namespace SAMRAI;

// Test that surface meters can compute when all points are inside the
// Cartesian grid correctly

void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<2>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);

  std::ofstream out;
  if (rank == 0)
    out.open("output");
  // test bottom face
  {
    std::vector<Point<2>> convex_hull;
    convex_hull.emplace_back(0.0, 0.0);
    convex_hull.emplace_back(0.5, 0.0);

    std::vector<Tensor<1, 2>> velocities(convex_hull.size());
    fdl::SurfaceMeter<2> meter_mesh(convex_hull,
                                    velocities,
                                    patch_hierarchy);

    const bool inside_domain = meter_mesh.compute_vertices_inside_domain();
    if (rank == 0)
      out << "meter has all points inside the domain = "
          << inside_domain
          << '\n';
  }

  // test top face
  {
    std::vector<Point<2>> convex_hull;
    convex_hull.emplace_back(0.0, 1.0);
    convex_hull.emplace_back(0.5, 1.0);

    std::vector<Tensor<1, 2>> velocities(convex_hull.size());
    fdl::SurfaceMeter<2> meter_mesh(convex_hull,
                                    velocities,
                                    patch_hierarchy);

    const bool inside_domain = meter_mesh.compute_vertices_inside_domain();
    if (rank == 0)
      out << "meter has all points inside the domain = "
          << inside_domain
          << '\n';
  }

  // test right face
  {
    std::vector<Point<2>> convex_hull;
    convex_hull.emplace_back(1.0, 0.5);
    convex_hull.emplace_back(1.0, 0.75);

    std::vector<Tensor<1, 2>> velocities(convex_hull.size());
    fdl::SurfaceMeter<2> meter_mesh(convex_hull,
                                    velocities,
                                    patch_hierarchy);

    const bool inside_domain = meter_mesh.compute_vertices_inside_domain();
    if (rank == 0)
      out << "meter has all points inside the domain = "
          << inside_domain
          << '\n';
  }

  // test inside domain
  {
    std::vector<Point<2>> convex_hull;
    convex_hull.emplace_back(0.5, 0.5);
    convex_hull.emplace_back(0.75, 0.75);

    std::vector<Tensor<1, 2>> velocities(convex_hull.size());
    fdl::SurfaceMeter<2> meter_mesh(convex_hull,
                                    velocities,
                                    patch_hierarchy);

    const bool inside_domain = meter_mesh.compute_vertices_inside_domain();
    if (rank == 0)
      out << "meter has all points inside the domain = "
          << inside_domain
          << '\n';
  }

  // test inside and outside domain
  {
    std::vector<Point<2>> convex_hull;
    convex_hull.emplace_back(0.5, 0.5);
    convex_hull.emplace_back(1.75, 1.75);

    std::vector<Tensor<1, 2>> velocities(convex_hull.size());
    fdl::SurfaceMeter<2> meter_mesh(convex_hull,
                                    velocities,
                                    patch_hierarchy);

    const bool inside_domain = meter_mesh.compute_vertices_inside_domain();
    if (rank == 0)
      out << "meter has all points inside the domain = "
          << inside_domain
          << '\n';
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test(app_initializer);
}
