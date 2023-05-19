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

  auto tuple = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);

  const unsigned int n_points = 8;
  std::vector<Point<spacedim>> boundary_points(n_points);
  for(unsigned int n = 0; n < n_points; ++n)
    {
      boundary_points[n][0] = std::cos(2.0 * numbers::PI * n / double(n_points));
      boundary_points[n][1] = std::sin(2.0 * numbers::PI * n / double(n_points));
      boundary_points[n][2] = std::sin(4.0 * numbers::PI * n / double(n_points));
    }

  std::vector<Tensor<1, spacedim>> velocities(n_points);

  fdl::SurfaceMeter<dim, spacedim> test_meter(boundary_points, velocities, patch_hierarchy);

  // const Triangulation<dim - 1, spacedim> &test_tria = test_meter.get_triangulation();
  const std::vector<Point<spacedim>> &vertex_points = test_meter.get_triangulation().get_vertices();

  std::ofstream output("output");
  for(unsigned int i = 0; i < n_points; ++i)
    {
      output << "original boundary point: " << boundary_points[i] << std::endl;
      output << "generated boundary point: " << vertex_points[i] << std::endl << std::endl;
    }
#if 0
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // plot native solution:
  {
    const Triangulation<dim - 1, spacedim> &meter_tria =
      meter_mesh.get_triangulation();
    DataOut<dim - 1, spacedim> data_out;
    data_out.attach_triangulation(meter_tria);
    if (rank == 0)
      output << "number of hull points = " << convex_hull.size() << std::endl
             << "number of vertices = " << meter_tria.get_vertices().size()
             << std::endl
             << "number of active cells = " << meter_tria.n_active_cells()
             << std::endl;

    Vector<float> subdomain(meter_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = meter_tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "solution", 0, mpi_comm, 2, 8);
  }
#endif
}

int main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv);

  test<3>(app_initializer);
}
