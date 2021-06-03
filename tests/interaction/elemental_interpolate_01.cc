#include <fiddle/base/exceptions.h>

#include <fiddle/interaction/elemental_interaction.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

#include "../tests.h"

// Test ElementalInteraction::compute_projection_rhs_*

using namespace dealii;
using namespace SAMRAI;

template <int spacedim, typename Number1, typename Number2>
dealii::BoundingBox<spacedim, Number1>
convert(const dealii::BoundingBox<spacedim, Number2> &input)
{
  // We should get a better conversion constructor
  dealii::Point<spacedim, Number1> p0;
  dealii::Point<spacedim, Number1> p1;
  for (unsigned int d = 0; d < spacedim; ++d)
    {
      p0[d] = input.get_boundary_points().first[d];
      p1[d] = input.get_boundary_points().second[d];
    }

  return dealii::BoundingBox<spacedim, Number1>(std::make_pair(p0, p1));
}

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto          input_db       = app_initializer->getInputDatabase();
  const int     n_F_components = get_n_f_components(input_db);
  constexpr int fe_degree      = 1;

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(mpi_comm);
  GridGenerator::concentric_hyper_shells(
    native_tria, Point<spacedim>(), 0.125, 0.25, 2, 0.0);
  native_tria.refine_global(4);

  std::unique_ptr<FiniteElement<dim>> F_fe;
  if (n_F_components == 1)
    F_fe = std::make_unique<FE_Q<dim>>(fe_degree);
  else
    F_fe =
      std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_degree), n_F_components);
  FESystem<dim> X_fe(FE_Q<dim>(fe_degree), dim);

  DoFHandler<dim> X_dof_handler(native_tria);
  X_dof_handler.distribute_dofs(X_fe);
  DoFHandler<dim> F_dof_handler(native_tria);
  F_dof_handler.distribute_dofs(*F_fe);
  IndexSet locally_relevant_X_dofs;
  DoFTools::extract_locally_relevant_dofs(X_dof_handler,
                                          locally_relevant_X_dofs);
  IndexSet locally_relevant_F_dofs;
  DoFTools::extract_locally_relevant_dofs(F_dof_handler,
                                          locally_relevant_F_dofs);

  auto X_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    X_dof_handler.locally_owned_dofs(),
    locally_relevant_X_dofs,
    native_tria.get_communicator());
  auto F_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    F_dof_handler.locally_owned_dofs(),
    locally_relevant_F_dofs,
    native_tria.get_communicator());

  MappingQ1<dim> F_mapping;
  QGauss<dim>    F_quadrature(fe_degree + 1);

  LinearAlgebra::distributed::Vector<double> X_vec(X_partitioner);
  VectorTools::interpolate(X_dof_handler,
                           Functions::IdentityFunction<dim>(),
                           X_vec);
  LinearAlgebra::distributed::Vector<double> F_rhs(F_partitioner);
  LinearAlgebra::distributed::Vector<double> F_solution(F_partitioner);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  // Now set up fiddle things for the test:
  std::vector<BoundingBox<spacedim, float>> bboxes;
  for (const auto &cell : native_tria.active_cell_iterators())
    if (cell->is_locally_owned())
      bboxes.emplace_back(
        convert<spacedim, float, double>(cell->bounding_box()));
  const auto all_bboxes =
    fdl::collect_all_active_cell_bboxes(native_tria, bboxes);

  fdl::ElementalInteraction<dim, spacedim> interaction(
    native_tria,
    all_bboxes,
    patch_hierarchy,
    patch_hierarchy->getFinestLevelNumber(),
    fe_degree + 1,
    1.0);
  interaction.add_dof_handler(X_dof_handler);
  interaction.add_dof_handler(F_dof_handler);

  // Do the test:
  auto transaction = interaction.compute_projection_rhs_start(
    f_idx, X_dof_handler, X_vec, F_dof_handler, F_mapping, F_rhs);
  transaction =
    interaction.compute_projection_rhs_intermediate(std::move(transaction));
  interaction.compute_projection_rhs_finish(std::move(transaction));

  {
    auto matrix_free = std::make_shared<MatrixFree<dim, double>>();
    AffineConstraints<double> constraints;
    constraints.close();
    matrix_free->reinit(F_mapping, F_dof_handler, constraints, F_quadrature);

    std::unique_ptr<MatrixFreeOperators::Base<dim>> mass_operator;
    using namespace MatrixFreeOperators;
    switch (n_F_components)
      {
        case 1:
          mass_operator =
            std::make_unique<MassOperator<dim, fe_degree, fe_degree + 1, 1>>();
          break;
        case 2:
          mass_operator =
            std::make_unique<MassOperator<dim, fe_degree, fe_degree + 1, 2>>();
          break;
        case 3:
          mass_operator =
            std::make_unique<MassOperator<dim, fe_degree, fe_degree + 1, 3>>();
          break;
        default:
          Assert(false, fdl::ExcFDLNotImplemented());
      }

    mass_operator->initialize(matrix_free);
    mass_operator->compute_diagonal();

    PreconditionIdentity preconditioner;
    preconditioner.initialize(*mass_operator);

    SolverControl control(100, 1e-14 * F_rhs.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
    cg.solve(*mass_operator, F_solution, F_rhs, preconditioner);
    F_solution.update_ghost_values();
  }

  // output:
  {
    auto              fp_db  = input_db->getDatabase("test")->getDatabase("f");
    const std::string x_vars = dim == 2 ? "X_0,X_1" : "X_0,X_1,X_2";
    const std::string fp_string = extract_fp_string(fp_db);
    FunctionParser<spacedim> fp(fp_string,
                                "PI=" + std::to_string(numbers::PI),
                                x_vars);
    Vector<double>           error(native_tria.n_active_cells());
    VectorTools::integrate_difference(
      F_dof_handler, F_solution, fp, error, F_quadrature, VectorTools::L2_norm);
    const double global_error =
      VectorTools::compute_global_error(native_tria,
                                        error,
                                        VectorTools::L2_norm);
    if (rank == 0)
      {
        std::ofstream output("output");
        output << "global error = " << global_error << std::endl;
      }
  }


  // plot deal.II:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(F_dof_handler);
    data_out.add_data_vector(F_solution, "F");

    Vector<float> subdomain(native_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = native_tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "solution", 0, mpi_comm, 2, 8);
  }

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<NDIM>(app_initializer);
}
