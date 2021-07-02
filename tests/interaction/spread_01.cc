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
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/SAMRAIGhostDataAccumulator.h>
#include <ibtk/muParserCartGridFunction.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <HierarchyCellDataOpsReal.h>
#include <HierarchySideDataOpsReal.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>

#include "../tests.h"

// Test spreading

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
  GridGenerator::hyper_cube(native_tria);
  // Even though we are periodic in both directions we don't ever need to
  // actually enforce this in the finite element code as far as spreading goes
  native_tria.refine_global(std::log2(input_db->getInteger("N")));

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  // Now set up fiddle things for the test:
  auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  for (auto &patch : patches)
    fdl::fill_all(patch->getPatchData(f_idx), 0.0);

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


  // set up what we need for spreading:
  const MappingQ<dim>                position_map(1);
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

  FunctionParser<spacedim> fp(
    extract_fp_string(input_db->getDatabase("test")->getDatabase("f")),
    "PI=" + std::to_string(numbers::PI),
    "X_0,X_1");

  Vector<double> F(F_dof_handler.n_dofs());
  VectorTools::interpolate(F_map, F_dof_handler, fp, F);

  fdl::compute_spread("BSPLINE_3",
                      f_idx,
                      patch_map,
                      position_map,
                      quadrature_indices,
                      quadratures,
                      F_dof_handler,
                      F_map,
                      F);

  // TODO - we need to accumulate data spread into ghost regions next
  SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<spacedim>> f_var;
  auto *var_db = hier::VariableDatabase<spacedim>::getDatabase();
  var_db->mapIndexToVariable(f_idx, f_var);
  // TODO un-hardcode ghost cell widths everywhere in the test suite
  const SAMRAI::hier::IntVector<spacedim> gcw(3);

  // TODO we should also accumulate forces spread outside the domain at this
  // point

  IBTK::SAMRAIGhostDataAccumulator acc(patch_hierarchy,
                                       f_var,
                                       gcw,
                                       patch_hierarchy->getFinestLevelNumber(),
                                       patch_hierarchy->getFinestLevelNumber());
  acc.accumulateGhostData(f_idx);

  // also compute the errors pointwise
  {
    using namespace SAMRAI;
    const int e_idx = var_db->registerClonedPatchDataIndex(f_var, f_idx);

    for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
      {
        tbox::Pointer<hier::PatchLevel<spacedim>> level =
          patch_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(e_idx, 0.0);
      }

    IBTK::muParserCartGridFunction f_fcn(
      "f",
      input_db->getDatabase("test")->getDatabase("f"),
      patch_hierarchy->getGridGeometry());
    f_fcn.setDataOnPatchHierarchy(e_idx, f_var, patch_hierarchy, 0.0);

    auto ops = fdl::extract_hierarchy_data_ops(f_var, patch_hierarchy);
    ops->subtract(e_idx, e_idx, f_idx);

    const double max_norm = ops->maxNorm(e_idx);
    if (rank == 0)
      {
        std::ofstream output("output");
        output << "Number of elements: " << native_tria.n_active_cells()
               << '\n';
        output << "max error = " << max_norm << '\n';
      }
  }

  // plot overlap data on each processor:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(F_dof_handler);
    data_out.add_data_vector(F, "F");
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
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
