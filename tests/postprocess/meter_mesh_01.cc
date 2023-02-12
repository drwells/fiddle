#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/meter_mesh.h>

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

// Test the meter mesh code for a basic interpolation problem

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  std::vector<Point<spacedim>> convex_hull;
  const unsigned int           n_points =
    4 * std::pow(test_db->getIntegerWithDefault("n_global_refinements", 4), 2);
  Assert(n_points > 1, fdl::ExcFDLInternalError());
  for (unsigned int p = 0; p < n_points; ++p)
    {
      convex_hull.emplace_back();
      convex_hull.back()[0] =
        0.9 * std::cos(2.0 * numbers::PI * p / double(n_points));
      convex_hull.back()[1] =
        0.9 * std::sin(2.0 * numbers::PI * p / double(n_points));
    }
  // make it a loop:
  convex_hull.emplace_back(convex_hull.front());
  std::vector<Tensor<1, dim>> velocities(convex_hull.size());

  fdl::MeterMesh<dim, spacedim> meter_mesh(convex_hull,
                                           velocities,
                                           patch_hierarchy);

  // do the actual test:
  const double interpolated_mean_value =
    meter_mesh.compute_mean_value(f_idx, "BSPLINE_3");
  double nodal_mean_value = 0.0;
  {
    FunctionParser<spacedim> fp(extract_fp_string(test_db->getDatabase("f")),
                                "PI=" + std::to_string(numbers::PI),
                                "X_0,X_1");

    for (std::size_t i = 0; i < convex_hull.size() - 1; ++i)
      nodal_mean_value += fp.value(convex_hull[i]);
    nodal_mean_value /= n_points;
  }

  std::ofstream output;
  if (rank == 0)
    {
      output.open("output");
      output << "interpolated mean value = " << interpolated_mean_value
             << std::endl;
      output << "nodal mean value = " << nodal_mean_value << std::endl;
    }

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

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
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
