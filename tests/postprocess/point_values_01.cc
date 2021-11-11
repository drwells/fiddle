#include <fiddle/postprocess/point_values.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

#include "../tests.h"

using namespace SAMRAI;
using namespace dealii;

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  const MPI_Comm mpi_comm = MPI_COMM_WORLD;
  auto           input_db = app_initializer->getInputDatabase();
  auto           test_db  = input_db->getDatabase("test");

  parallel::shared::Triangulation<dim, spacedim> tria(
    mpi_comm, {}, test_db->getBoolWithDefault("use_artificial_cells", false));
  GridGenerator::hyper_cube(tria);
  std::vector<Point<spacedim>> evaluation_points;
  {
    auto cell = tria.begin_active();
    for (unsigned int vertex_n : cell->vertex_indices())
      evaluation_points.push_back(cell->vertex(vertex_n));
  }
  tria.refine_global(3);

  std::ostringstream local_out;

  const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
  if (rank != 0)
    local_out << '\n';
  local_out << "rank = " << rank << '\n';

  {
    local_out << "Test with vector-valued FE\n";
    FESystem<dim, spacedim>   fe(FE_Q<dim, spacedim>(1), spacedim);
    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
    LinearAlgebra::distributed::Vector<double> solution(partitioner);
    VectorTools::interpolate(dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             solution);

    MappingQ<dim, spacedim>                   mapping(1);
    fdl::PointValues<spacedim, dim, spacedim> point_values(mapping,
                                                           dof_handler,
                                                           evaluation_points);
    const auto result = point_values.evaluate(solution);

    for (std::size_t i = 0; i < result.size(); ++i)
      local_out << evaluation_points[i] << ", " << result[i] << '\n';
  }

  {
    local_out << "\nTest with scalar-valued FE\n";
    FE_Q<dim, spacedim>       fe(2);
    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
    LinearAlgebra::distributed::Vector<double> solution(partitioner);
    VectorTools::interpolate(dof_handler,
                             Functions::CosineFunction<spacedim>(),
                             solution);

    MappingQ<dim, spacedim>            mapping(1);
    fdl::PointValues<1, dim, spacedim> point_values(mapping,
                                                    dof_handler,
                                                    evaluation_points);
    const auto                         result = point_values.evaluate(solution);

    for (std::size_t i = 0; i < result.size(); ++i)
      local_out << evaluation_points[i] << ", " << result[i] << '\n';
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
    new IBTK::AppInitializer(argc, argv, "point_values_01.log");

  test<NDIM>(app_initializer);
}
