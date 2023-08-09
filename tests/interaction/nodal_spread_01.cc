#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/nodal_patch_map.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/matrix_free/operators.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_rhs.h>

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

// Test nodal spreading - in particular, see if nodes on (SAMRAI) cell
// boundaries get spread from twice. This might be mitigated by the 'bug' in
// SAMRAI in which x_upper of the lower patch doesn't match x_lower of the upper
// patch.

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
  const auto partitioner =
    parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder;
  parallel::shared::Triangulation<dim, spacedim> native_tria(mpi_comm,
                                                             {},
                                                             false,
                                                             partitioner);
  GridGenerator::hyper_cube(native_tria);
  // Even though we are periodic in both directions we don't ever need to
  // actually enforce this in the finite element code as far as spreading goes
  native_tria.refine_global(std::log2(input_db->getInteger("N")));
  // For uniqueness, points on the upper boundaries of (SAMRAI) cells are
  // assigned to be on the higher-indexed cell. This is problematic in our case
  // since we have DoFs right on the upper boundary. Get around this by shifting
  // the boundary nodes very slightly.
  for (auto &cell : native_tria.active_cell_iterators())
    for (const unsigned int vertex_no : cell->vertex_indices())
      for (unsigned int d = 0; d < spacedim; ++d)
        if (cell->vertex(vertex_no)[d] == 1.0)
          cell->vertex(vertex_no)[d] = std::nexttoward(1.0, 0.0);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  // Now set up deal.II:
  const auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  for (auto &patch : patches)
    fdl::fill_all(patch->getPatchData(f_idx), 0.0);

  const int n_F_components = get_n_f_components(input_db);
  std::unique_ptr<FiniteElement<dim>> fe;
  if (n_F_components == 1)
    {
      fe = std::make_unique<FE_Q<dim>>(1);
    }
  else
    {
      fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(1), n_F_components);
    }

  DoFHandler<dim, spacedim> dof_handler(native_tria);
  dof_handler.distribute_dofs(*fe);
  DoFRenumbering::support_point_wise(dof_handler);
  const MappingQ<dim, spacedim> mapping(1);

  FunctionParser<spacedim> fp(
    extract_fp_string(input_db->getDatabase("test")->getDatabase("f")),
    "PI=" + std::to_string(numbers::PI),
    "X_0,X_1");

  Vector<double> spread_values(dof_handler.n_dofs());
  VectorTools::create_right_hand_side(mapping,
                                      dof_handler,
                                      QGauss<dim>(fe->tensor_degree() + 1),
                                      fp,
                                      spread_values);

  Vector<double> position(dof_handler.n_dofs());
  VectorTools::interpolate(mapping,
                           dof_handler,
                           Functions::IdentityFunction<spacedim>(),
                           position);

  // and fiddle:
  std::vector<std::vector<BoundingBox<spacedim>>> bboxes;
  {
    const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geometry =
      patches.back()->getPatchGeometry();
    Assert(geometry, fdl::ExcFDLNotImplemented());
    const double *const patch_dx = geometry->getDx();
    for (const auto &patch : patches)
      {
        bboxes.emplace_back();
        bboxes.back().push_back(
          fdl::box_to_bbox(patch->getBox(),
                           patch_hierarchy->getPatchLevel(
                             patch_hierarchy->getFinestLevelNumber())));
        bboxes.back().back().extend(1.0 * patch_dx[0]);
      }
  }
  fdl::NodalPatchMap<dim, spacedim> nodal_patch_map(patches, bboxes, position);

  fdl::compute_nodal_spread(
    "BSPLINE_3", f_idx, nodal_patch_map, position, spread_values);

  SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<spacedim>> f_var;
  auto *var_db = hier::VariableDatabase<spacedim>::getDatabase();
  var_db->mapIndexToVariable(f_idx, f_var);
  // TODO un-hardcode ghost cell widths everywhere in the test suite
  const SAMRAI::hier::IntVector<spacedim> gcw(3);

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

        output << std::setprecision(16);
        for (const auto &patch : patches)
          {
            tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> patch_geom =
              patch->getPatchGeometry();
            const auto xlower = patch_geom->getXLower();
            const auto xupper = patch_geom->getXUpper();
            output << "patch\n";
            output << "patch lower = ";
            for (unsigned int d = 0; d < spacedim - 1; ++d)
              output << xlower[d] << ", ";
            output << xlower[spacedim - 1] << '\n';
            output << "patch upper = ";
            for (unsigned int d = 0; d < spacedim - 1; ++d)
              output << xupper[d] << ", ";
            output << xupper[spacedim - 1] << '\n';
          }
      }
  }

  // plot overlap data on each processor:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(spread_values, "spread_values");
    data_out.build_patches();
    std::ofstream data_out_stream("spread-values-" + std::to_string(rank) +
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
