#include <fiddle/base/exceptions.h>

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>

#include "../tests.h"

// Test velocity interpolation

using namespace SAMRAI;
using namespace dealii;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD);
  GridGenerator::concentric_hyper_shells(
    native_tria, Point<spacedim>(), 0.125, 0.25, 2, 0.0);
  native_tria.refine_global(4);

  // setup SAMRAI stuff (its always the same):
  auto pair            = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = pair.first;
  auto u_cc_idx        = pair.second;

  // Now set up fiddle things for the test:
  const auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  const std::vector<BoundingBox<spacedim>> patch_bboxes =
    fdl::compute_patch_bboxes(patches, 1.0);
  fdl::TriaIntersectionPredicate<spacedim> tria_pred(patch_bboxes);
  fdl::OverlapTriangulation<spacedim>      overlap_tria(native_tria, tria_pred);
  std::vector<BoundingBox<spacedim, float>> cell_bboxes;
  for (const auto &cell : overlap_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      cell_bboxes.push_back(fbbox);
    }
  fdl::PatchMap<dim, spacedim> patch_map(patches,
                                         1.0,
                                         overlap_tria,
                                         cell_bboxes);

  std::ofstream output;
  if (rank == 0)
    output.open("output");

  // set up what we need for computing the RHS of the projection:
  const MappingQ<dim>                X_map(1);
  const std::vector<Quadrature<dim>> quadratures({QGauss<dim>(2)});
  const std::vector<unsigned char>   quadrature_indices(
    overlap_tria.n_active_cells());

  const int n_F_components = get_n_f_components(input_db);
  // TODO - it would be nice to make this work with n_F_components = 1, but that
  // messes up some MatrixFree implementation details
  std::unique_ptr<FiniteElement<dim>> fe;
  if (n_F_components == 1)
    {
      fe = std::make_unique<FE_Q<dim>>(1);
    }
  else
    {
      fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(1), n_F_components);
    }

  DoFHandler<dim, spacedim> F_dof_handler(overlap_tria);
  F_dof_handler.distribute_dofs(*fe);
  const MappingQ<dim, spacedim> F_map(1);
  Vector<double>                F_rhs(F_dof_handler.n_dofs());

  compute_projection_rhs(u_cc_idx,
                         patch_map,
                         X_map,
                         quadrature_indices,
                         quadratures,
                         F_dof_handler,
                         F_map,
                         F_rhs);

  Vector<double> F_solution = F_rhs;
  // Do the projection locally (this is just a test)
  {
    DynamicSparsityPattern dsp(F_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(F_dof_handler, dsp);
    SparsityPattern sp;
    sp.copy_from(dsp);
    SparseMatrix<double> mass_matrix(sp);

    MatrixCreator::create_mass_matrix(F_map,
                                      F_dof_handler,
                                      quadratures[0],
                                      mass_matrix);

    PreconditionIdentity preconditioner;
    preconditioner.initialize(mass_matrix);

    SolverControl control(100, 1e-14 * F_rhs.l2_norm());
    SolverCG<>    cg(control);
    cg.solve(mass_matrix, F_solution, F_rhs, preconditioner);
  }

  // scatter the overlap data back to the native data and solve the native
  // system:
  DoFHandler<dim, spacedim> F_native_dof_handler(native_tria);
  F_native_dof_handler.distribute_dofs(*fe);
  LinearAlgebra::distributed::Vector<double> native_solution, native_rhs;
  {
    const std::vector<types::global_dof_index> native_dofs =
      fdl::compute_overlap_to_native_dof_translation(overlap_tria,
                                                     F_dof_handler,
                                                     F_native_dof_handler);
    auto matrix_free = std::make_shared<MatrixFree<dim, double>>();
    // the canned mass operator needs a shared pointer
    AffineConstraints<double> constraints;
    constraints.close();
    matrix_free->reinit(F_map,
                        F_native_dof_handler,
                        constraints,
                        quadratures[0]);
    matrix_free->initialize_dof_vector(native_solution);
    matrix_free->initialize_dof_vector(native_rhs);

    fdl::Scatter<double> scatter(native_dofs,
                                 F_native_dof_handler.locally_owned_dofs(),
                                 mpi_comm);

    scatter.overlap_to_global_start(F_rhs, VectorOperation::add, 0, native_rhs);
    scatter.overlap_to_global_finish(F_rhs, VectorOperation::add, native_rhs);

    std::unique_ptr<MatrixFreeOperators::Base<dim>> mass_operator;
    // TODO - we need some utility functions for converting input parameters to
    // template parameters
    switch (n_F_components)
      {
        case 1:
          mass_operator =
            std::make_unique<MatrixFreeOperators::MassOperator<dim, 1, 2, 1>>();
          break;
        case 2:
          mass_operator =
            std::make_unique<MatrixFreeOperators::MassOperator<dim, 1, 2, 2>>();
          break;
        case 3:
          mass_operator =
            std::make_unique<MatrixFreeOperators::MassOperator<dim, 1, 2, 3>>();
          break;
        default:
          Assert(false, fdl::ExcFDLNotImplemented());
      }

    mass_operator->initialize(matrix_free);
    mass_operator->compute_diagonal();

    PreconditionIdentity preconditioner;
    preconditioner.initialize(*mass_operator);

    SolverControl control(100, 1e-14 * native_rhs.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
    cg.solve(*mass_operator, native_solution, native_rhs, preconditioner);
  }

  // plot overlap data on each processor:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(F_dof_handler);
    data_out.add_data_vector(F_solution, "F_solution");
    data_out.add_data_vector(F_rhs, "F_rhs");
    data_out.build_patches();
    std::ofstream data_out_stream("overlap-tria-" + std::to_string(rank) +
                                  ".vtu");
    data_out.write_vtu(data_out_stream);
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
    native_data_out.attach_dof_handler(F_native_dof_handler);
    native_solution.update_ghost_values();
    native_data_out.add_data_vector(native_solution, "F");

    Vector<float> subdomain(native_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = native_tria.locally_owned_subdomain();
    native_data_out.add_data_vector(subdomain, "subdomain");

    native_data_out.build_patches();

    native_data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_comm, 2, 8);
  }

  // save output:
  {
    auto        fp_db = input_db->getDatabase("test")->getDatabase("f");
    std::string fp_string;
    if (n_F_components == 1)
      {
        fp_string += fp_db->getString("function");
      }
    else
      {
        for (int c = 0; c < n_F_components; ++c)
          {
            fp_string += fp_db->getString("function_" + std::to_string(c));
            if (c != n_F_components - 1)
              fp_string += ';';
          }
      }

    FunctionParser<spacedim> fp(fp_string,
                                "PI=" + std::to_string(numbers::PI),
                                "X_0,X_1");
    Vector<float>            error(native_tria.n_active_cells());
    VectorTools::integrate_difference(F_native_dof_handler,
                                      native_solution,
                                      fp,
                                      error,
                                      quadratures[0],
                                      VectorTools::L2_norm);
    const double global_error =
      VectorTools::compute_global_error(native_tria,
                                        error,
                                        VectorTools::L2_norm);
    if (rank == 0)
      output << "global error = " << global_error << std::endl;
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
