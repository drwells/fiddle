#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/volume_meter.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

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

// Like meter_mesh_01 but for VolumeMeter

template <int dim, int spacedim = dim>
void
test(
  SAMRAI::tbox::Pointer<IBTK::AppInitializer>                   app_initializer,
  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<spacedim>> patch_hierarchy,
  const int                                                     f_idx,
  const bool use_tria_ctor = false)
{
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;

  Point<spacedim>            center;
  fdl::VolumeMeter<spacedim> meter_mesh(center, 4, patch_hierarchy);

  // do the actual test:
  double interpolated_mean_value =
    meter_mesh.compute_mean_value(f_idx, "BSPLINE_3");

  tbox::plog << "centroid = " << meter_mesh.get_centroid() << std::endl
             << "interpolated mean value centered at " << center << " = "
             << interpolated_mean_value << std::endl;

  for (unsigned int d = 0; d < spacedim; ++d)
    center[d] = 0.1;
  meter_mesh.reinit(center);

  interpolated_mean_value = meter_mesh.compute_mean_value(f_idx, "BSPLINE_3");
  tbox::plog << "centroid = " << meter_mesh.get_centroid() << std::endl
             << "interpolated mean value centered at " << center << " = "
             << interpolated_mean_value << std::endl;

  // write SAMRAI data:
  {
    // samrai won't let us write the same thing twice
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         int(use_tria_ctor),
                                                         double(use_tria_ctor));
  }

  // plot native solution:
  {
    const Triangulation<spacedim> &meter_tria = meter_mesh.get_triangulation();
    DataOut<spacedim>              data_out;
    data_out.attach_triangulation(meter_tria);
    tbox::plog << "number of vertices = " << meter_tria.get_vertices().size()
               << std::endl
               << "number of active cells = " << meter_tria.n_active_cells()
               << std::endl;

    const LinearAlgebra::distributed::Vector<double> interpolated_f =
      meter_mesh.interpolate_scalar_field(f_idx, "BSPLINE_3");
    data_out.attach_dof_handler(meter_mesh.get_scalar_dof_handler());
    data_out.add_data_vector(interpolated_f,
                             "f",
                             decltype(data_out)::DataVectorType::type_dof_data);

    Vector<float> subdomain(meter_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = meter_tria.locally_owned_subdomain();
    data_out.add_data_vector(
      subdomain,
      "subdomain",
      decltype(data_out)::DataVectorType::type_cell_data);

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", unsigned(use_tria_ctor), mpi_comm, 2, 8);
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<2>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  test<2>(app_initializer, patch_hierarchy, f_idx, false);
  test<2>(app_initializer, patch_hierarchy, f_idx, true);
}
