#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/nodal_patch_map.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>
#include <random>

#include "../tests.h"

// Test nodal interpolation without using FEM (just the utility functions)

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD);
  GridGenerator::concentric_hyper_shells(
    native_tria, Point<spacedim>(), 0.125, 0.25, 2, 0.0);
  native_tria.refine_global(4);

  // setup SAMRAI stuff (its always the same):
  auto      tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto      patch_hierarchy = std::get<0>(tuple);
  auto      f_idx           = std::get<5>(tuple);
  const int n_F_components  = get_n_f_components(input_db);

  // setup Lagrangian data:
  const std::size_t n_nodes = test_db->getIntegerWithDefault("n_nodes", 10);
  Vector<double>    nodal_coordinates(n_nodes * spacedim);
  Vector<double>    interpolated_values(n_nodes * n_F_components);
  std::mt19937      std_seq(42u);
  std::uniform_real_distribution<double> distribution(0.1, 0.9);
  for (double &coordinate : nodal_coordinates)
    coordinate = distribution(std_seq);

  // Now set up fiddle things for the test:
  const auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geometry =
    patches.back()->getPatchGeometry();
  Assert(geometry, fdl::ExcFDLNotImplemented());
  const double *const                             patch_dx = geometry->getDx();
  std::vector<std::vector<BoundingBox<spacedim>>> bboxes;
  for (const auto &patch : patches)
    {
      bboxes.emplace_back();
      bboxes.back().push_back(
        fdl::box_to_bbox(patch->getBox(),
                         patch_hierarchy->getPatchLevel(
                           patch_hierarchy->getFinestLevelNumber())));
      bboxes.back().back().extend(1.0 * patch_dx[0]);
    }

  fdl::NodalPatchMap<dim, spacedim> patch_map(patches,
                                              bboxes,
                                              nodal_coordinates);

  // Actual test:
  std::ofstream output;
  if (rank == 0)
    output.open("output");

  compute_nodal_interpolation(
    "BSPLINE_3", f_idx, patch_map, nodal_coordinates, interpolated_values);

  for (std::size_t node_n = 0; node_n < n_nodes; ++node_n)
    {
      for (unsigned int d = 0; d < spacedim; ++d)
        output << nodal_coordinates[spacedim * node_n + d] << ", ";

      for (int c = 0; c < n_F_components - 1; ++c)
        output << interpolated_values[n_F_components * node_n + c] << ", ";
      output
        << interpolated_values[n_F_components * node_n + n_F_components - 1]
        << '\n';
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
