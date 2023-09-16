#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/intersection_predicate_lib.h>
#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/matrix_free/operators.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>

#include "../tests.h"

// Test cell tagging

using namespace dealii;
using namespace SAMRAI;

template <int spacedim>
class TestTag : public mesh::StandardTagAndInitStrategy<spacedim>
{
public:
  TestTag(const std::vector<BoundingBox<spacedim, float>> &bboxes)
    : cell_bboxes(bboxes)
  {}

  virtual void
  initializeLevelData(
    const tbox::Pointer<hier::BasePatchHierarchy<spacedim>> /*hierarchy*/,
    const int /*level_number*/,
    const double /*init_data_time*/,
    const bool /*can_be_refined*/,
    const bool /*initial_time*/,
    const tbox::Pointer<hier::BasePatchLevel<spacedim>> /*old_level*/ = nullptr,
    const bool /*allocate_data*/ = true) override
  {}

  virtual void
  resetHierarchyConfiguration(
    const tbox::Pointer<hier::BasePatchHierarchy<spacedim>> /*hierarchy*/,
    const int /*coarsest_level*/,
    const int /*finest_level*/) override
  {}

  virtual void
  applyGradientDetector(
    const tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
    const int                                               level_number,
    const double /*error_data_time*/,
    const int tag_index,
    const bool /*initial_time*/,
    const bool /*uses_richardson_extrapolation_too*/) override
  {
    tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
      hierarchy->getPatchLevel(level_number);
    fdl::tag_cells(cell_bboxes, tag_index, patch_level);
  }

protected:
  std::vector<BoundingBox<spacedim, float>> cell_bboxes;
};


template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  std::ofstream timer_out("timer-out-" + std::to_string(rank) + ".txt");
  TimerOutput   computing_timer(MPI_COMM_SELF,
                              timer_out,
                              TimerOutput::summary,
                              TimerOutput::wall_times);
  // setup deal.II stuff:
  const auto partitioner =
    parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder;
  parallel::shared::Triangulation<dim, spacedim> native_tria(mpi_comm,
                                                             {},
                                                             false,
                                                             partitioner);
  {
    TimerOutput::Scope t(computing_timer, "grid_in");
    GridIn<dim>        grid_in(native_tria);
    grid_in.read_exodusii("whole_heart_with_fibers_v5.e");
  }

  // test:
  std::vector<BoundingBox<spacedim, float>> all_cell_bboxes;
  {
    TimerOutput::Scope t(computing_timer, "all_bboxes");
    for (const auto &cell : native_tria.active_cell_iterators())
      {
        BoundingBox<spacedim, float> fbbox;
        fbbox.get_boundary_points() =
          cell->bounding_box().get_boundary_points();
        all_cell_bboxes.push_back(fbbox);
      }
  }
  TestTag<spacedim> test_tag(all_cell_bboxes);

  // setup SAMRAI stuff (its always the same)
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);


  // Now set up fiddle things for the test:
  const auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  const std::vector<BoundingBox<spacedim>> patch_bboxes =
    fdl::compute_patch_bboxes(patches, 1.0);
  fdl::TriaIntersectionPredicate<spacedim> tria_pred(patch_bboxes);
  fdl::OverlapTriangulation<spacedim>      overlap_tria(native_tria, tria_pred);

  std::vector<BoundingBox<spacedim, float>> overlap_cell_bboxes;
  for (const auto &cell : overlap_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      overlap_cell_bboxes.push_back(fbbox);
    }

  TimerOutput::Scope           pmt(computing_timer, "setup_patch_map");
  fdl::PatchMap<dim, spacedim> patch_map(patches,
                                         1.0,
                                         overlap_tria,
                                         overlap_cell_bboxes);
  pmt.stop();

  std::ofstream output;
  if (rank == 0)
    output.open("output");

  const Mapping<dim, spacedim> &position_map =
    get_default_linear_mapping<dim, spacedim>(overlap_tria);
  std::vector<Quadrature<dim>> quadratures;
  quadratures.push_back(QGaussSimplex<dim>(2));
  std::vector<unsigned char> quadrature_indices(overlap_tria.n_active_cells());

  FE_SimplexP<dim>          fe(1);
  DoFHandler<dim, spacedim> F_dof_handler(overlap_tria);
  F_dof_handler.distribute_dofs(fe);
  const Mapping<dim, spacedim> &F_map =
    get_default_linear_mapping<dim, spacedim>(overlap_tria);
  Vector<double> F_rhs(F_dof_handler.n_dofs());

  {
    TimerOutput::Scope cprt(computing_timer, "compute_projection_rhs");
    compute_projection_rhs("BSPLINE_3",
                           f_idx,
                           patch_map,
                           position_map,
                           quadrature_indices,
                           quadratures,
                           F_dof_handler,
                           F_map,
                           F_rhs);
  }

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
    SparseDirectUMFPACK sparse_direct;
    sparse_direct.initialize(mass_matrix);
    sparse_direct.solve(F_solution);
  }

  // also move things back to the actual system.
  DoFHandler<dim, spacedim> F_native_dof_handler(native_tria);
  F_native_dof_handler.distribute_dofs(fe);
  LinearAlgebra::distributed::Vector<double> native_solution, native_rhs;

  {
    TimerOutput::Scope cdt(computing_timer, "compute_dof_translation");
    const std::vector<types::global_dof_index> native_dofs =
      fdl::compute_overlap_to_native_dof_translation(overlap_tria,
                                                     F_dof_handler,
                                                     F_native_dof_handler);
    cdt.stop();

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

    {
      TimerOutput::Scope sotgt(computing_timer, "scatter_overlap_to_global");
      scatter.overlap_to_global_start(F_rhs,
                                      VectorOperation::add,
                                      0,
                                      native_rhs);
      scatter.overlap_to_global_finish(F_rhs, VectorOperation::add, native_rhs);
    }

    MatrixFreeOperators::MassOperator<dim, 1, 2, 1> mass_operator;
    mass_operator.initialize(matrix_free);
    mass_operator.compute_diagonal();

    PreconditionJacobi<decltype(mass_operator)> preconditioner;
    preconditioner.initialize(mass_operator);

    TimerOutput::Scope st(computing_timer, "solve");
    SolverControl      control(100, 1e-10);
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
    cg.solve(mass_operator, native_solution, native_rhs, preconditioner);
  }


  // plot overlap data on each processor:
  DataOut<dim> data_out;
  data_out.attach_dof_handler(F_dof_handler);
  data_out.add_data_vector(F_solution, "F_solution");
  data_out.add_data_vector(F_rhs, "F_rhs");
  data_out.build_patches();
  std::ofstream data_out_stream("overlap-tria-" + std::to_string(rank) +
                                ".vtu");
  data_out.write_vtu(data_out_stream);

  app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy, 0, 0.0);

  // plot native solution
  {
    DataOut<dim> native_data_out;
    native_data_out.attach_dof_handler(F_native_dof_handler);
    native_solution.update_ghost_values();
    native_data_out.add_data_vector(native_solution, "u");

    Vector<float> subdomain(native_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = native_tria.locally_owned_subdomain();
    native_data_out.add_data_vector(subdomain, "subdomain");

    native_data_out.build_patches();

    // The next step is to write this data to disk. We write up to 8 VTU files
    // in parallel with the help of MPI-IO. Additionally a PVTU record is
    // generated, which groups the written VTU files.
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

  test<3>(app_initializer);
}
