#include <fiddle/base/exceptions.h>

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/interaction/nodal_interaction.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

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

// Test multilevel nodal interpolation

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(mpi_comm);
  GridGenerator::hyper_ball(native_tria, Point<spacedim>(), 1.0);
  const auto test_db = input_db->getDatabase("test");
  native_tria.refine_global(
    test_db->getIntegerWithDefault("n_global_refinements", 4));

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  FESystem<dim>             position_fe(FE_Q<dim>(1), dim);
  DoFHandler<dim, spacedim> position_dof_handler(native_tria);
  position_dof_handler.distribute_dofs(position_fe);

  IndexSet locally_relevant_position_dofs;
  DoFTools::extract_locally_relevant_dofs(position_dof_handler,
                                          locally_relevant_position_dofs);
  auto position_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    position_dof_handler.locally_owned_dofs(),
    locally_relevant_position_dofs,
    mpi_comm);
  LinearAlgebra::distributed::Vector<double> position(position_partitioner);
  VectorTools::interpolate(position_dof_handler,
                           Functions::IdentityFunction<spacedim>(),
                           position);
  position.update_ghost_values();

  std::unique_ptr<FiniteElement<dim, spacedim>> F_fe;
  const unsigned int f_degree = test_db->getIntegerWithDefault("f_degree", 1);
  const int n_F_components = test_db->getDatabase("f")->getAllKeys().getSize();
  if (test_db->getBoolWithDefault("discontinuous_fe", false) == true)
    {
      if (n_F_components > 1)
        F_fe = std::make_unique<FESystem<dim, spacedim>>(
          FE_DGQ<dim, spacedim>(f_degree), n_F_components);
      else
        F_fe = std::make_unique<FE_DGQ<dim, spacedim>>(f_degree);
    }
  else
    {
      if (n_F_components > 1)
        F_fe = std::make_unique<FESystem<dim, spacedim>>(
          FE_Q<dim, spacedim>(f_degree), n_F_components);
      else
        F_fe = std::make_unique<FE_Q<dim, spacedim>>(f_degree);
    }
  DoFHandler<dim, spacedim> F_dof_handler(native_tria);
  F_dof_handler.distribute_dofs(*F_fe);
  const MappingQ<dim> F_mapping(1);

  IndexSet locally_relevant_F_dofs;
  DoFTools::extract_locally_relevant_dofs(F_dof_handler,
                                          locally_relevant_F_dofs);
  auto F_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    F_dof_handler.locally_owned_dofs(), locally_relevant_F_dofs, mpi_comm);
  LinearAlgebra::distributed::Vector<double> interpolated_F(F_partitioner);

  // Now set up fiddle things for the test:
  std::vector<BoundingBox<spacedim, float>> cell_bboxes;
  for (const auto &cell : native_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      cell_bboxes.push_back(fbbox);
    }

  fdl::NodalInteraction<dim, spacedim> interaction(
    input_db,
    native_tria,
    cell_bboxes,
    patch_hierarchy,
    std::make_pair(0, patch_hierarchy->getFinestLevelNumber()),
    position_dof_handler,
    position);

  // do the actual test:
  double global_error = 0.0;
  {
    interaction.add_dof_handler(F_dof_handler);
    auto transaction =
      interaction.compute_projection_rhs_start("BSPLINE_3",
                                               f_idx,
                                               position_dof_handler,
                                               position,
                                               F_dof_handler,
                                               F_mapping,
                                               interpolated_F);
    transaction =
      interaction.compute_projection_rhs_intermediate(std::move(transaction));
    interaction.compute_projection_rhs_finish(std::move(transaction));

    FunctionParser<spacedim> fp(extract_fp_string(test_db->getDatabase("f")),
                                "PI=" + std::to_string(numbers::PI),
                                "X_0,X_1");
    Vector<float>            error(native_tria.n_active_cells());
    interpolated_F.update_ghost_values();
    VectorTools::integrate_difference(F_dof_handler,
                                      interpolated_F,
                                      fp,
                                      error,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    global_error = VectorTools::compute_global_error(native_tria,
                                                     error,
                                                     VectorTools::L2_norm);
  }

  if (rank == 0)
    {
      std::ofstream output("output");
      output << "global L2 error = " << global_error << std::endl;
    }

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  // plot native solution:
  {
    DataOut<dim> native_data_out;
    native_data_out.attach_dof_handler(F_dof_handler);
    native_data_out.add_data_vector(interpolated_F, "F");

    Vector<float> subdomain(native_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = native_tria.locally_owned_subdomain();
    native_data_out.add_data_vector(subdomain, "subdomain");

    native_data_out.build_patches();

    native_data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_comm, 2, 8);
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
