#include <fiddle/grid/box_utilities.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>

#include "../tests.h"

// Test compute_nonoverlapping_patch_boxes()

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  std::ostringstream output;

  // Set up basic SAMRAI stuff:
  tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
    new geom::CartesianGridGeometry<NDIM>("CartesianGeometry",
                                          app_initializer->getComponentDatabase(
                                            "CartesianGeometry"));
  tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy =
    new hier::PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
  tbox::Pointer<mesh::StandardTagAndInitialize<NDIM>> error_detector =
    new mesh::StandardTagAndInitialize<NDIM>(
      "StandardTagAndInitialize",
      nullptr,
      app_initializer->getComponentDatabase("StandardTagAndInitialize"));

  tbox::Pointer<mesh::BergerRigoutsos<NDIM>> box_generator =
    new mesh::BergerRigoutsos<NDIM>();
  tbox::Pointer<mesh::LoadBalancer<NDIM>> load_balancer =
    new mesh::LoadBalancer<NDIM>(
      "LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
  tbox::Pointer<mesh::GriddingAlgorithm<NDIM>> gridding_algorithm =
    new mesh::GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                      app_initializer->getComponentDatabase(
                                        "GriddingAlgorithm"),
                                      error_detector,
                                      box_generator,
                                      load_balancer);

  // set up the SAMRAI grid:
  gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
  int level_number = 0;
  while (gridding_algorithm->levelCanBeRefined(level_number))
    {
      gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, 1);
      ++level_number;
    }

  // Set up a variable so that we can actually output the grid:
  auto *var_db = hier::VariableDatabase<NDIM>::getDatabase();
  tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");
  tbox::Pointer<pdat::CellVariable<spacedim, double>> u_cc_var =
    new pdat::CellVariable<spacedim, double>("u_cc");
  const int u_cc_idx =
    var_db->registerVariableAndContext(u_cc_var,
                                       ctx,
                                       hier::IntVector<spacedim>(1));

  const int finest_level = patch_hierarchy->getFinestLevelNumber();
  for (int ln = 0; ln <= finest_level; ++ln)
    {
      tbox::Pointer<hier::PatchLevel<spacedim>> level =
        patch_hierarchy->getPatchLevel(ln);
      level->allocatePatchData(u_cc_idx, 0.0);

      // obviously this won't generalize well
      auto patches = fdl::extract_patches(level);
      for (auto &patch : patches)
        {
          tbox::Pointer<pdat::CellData<spacedim, double>> data =
            patch->getPatchData(u_cc_idx);
          Assert(data, ExcMessage("pointer should not be null"));
          data->fillAll(0.0);
        }
    }

  // setup visualization:
  auto visit_data_writer = app_initializer->getVisItDataWriter();
  TBOX_ASSERT(visit_data_writer);
  visit_data_writer->registerPlotQuantity(u_cc_var->getName(),
                                          "SCALAR",
                                          u_cc_idx);
  visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);

  // save test output:
  print_partitioning_on_0(patch_hierarchy, 0, finest_level, output);

  std::vector<std::vector<hier::Box<spacedim>>> boxes =
    fdl::compute_nonoverlapping_patch_boxes(patch_hierarchy->getPatchLevel(0),
                                            patch_hierarchy->getPatchLevel(1));

  const int     rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  output << "rank = " << rank << std::endl;
  for (const std::vector<hier::Box<spacedim>> &patch_boxes : boxes)
    {
      if (patch_boxes.size() == 0)
        output << "  Box has an empty intersection\n";
      else
        {
          output << "  boxes =";
          for (const hier::Box<spacedim> &box : patch_boxes)
            output << " " << box;
          output << '\n';
        }
    }


  std::ofstream file_output;
  if (rank == 0)
    file_output.open("output");
  print_strings_on_0(output.str(), MPI_COMM_WORLD, file_output);
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
