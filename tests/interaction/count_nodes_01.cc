#include <fiddle/base/exceptions.h>

#include <fiddle/grid/nodal_patch_map.h>
#include <fiddle/grid/overlap_tria.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <HierarchyCellDataOpsReal.h>
#include <HierarchySideDataOpsReal.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>
#include <random>

#include "../tests.h"

// Test count_nodes

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
  const auto n_procs  = Utilities::MPI::n_mpi_processes(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_ball(native_tria);
  // Even though we are periodic in both directions we don't ever need to
  // actually enforce this in the finite element code as far as spreading goes
  native_tria.refine_global(std::log2(input_db->getInteger("N") / 2));

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  // setup Lagrangian data:
  const std::size_t n_nodes = test_db->getIntegerWithDefault("n_nodes", 10);
  Vector<double>    nodal_coordinates(n_nodes * spacedim);
  std::mt19937      std_seq(42u);
  std::uniform_real_distribution<double> distribution(0.3, 0.7);
  for (double &coordinate : nodal_coordinates)
    coordinate = distribution(std_seq);

  // Now set up fiddle things for the test:
  const auto patches = fdl::extract_patches(
    patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber()));
  for (auto &patch : patches)
    fdl::fill_all(patch->getPatchData(f_idx), 0.0);
  fdl::NodalPatchMap<dim, spacedim> nodal_patch_map(patches,
                                                    1.0,
                                                    nodal_coordinates);

  fdl::count_nodes(f_idx, nodal_patch_map, nodal_coordinates);

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  // write test output file:
  std::ostringstream out;
  {
    const int ln = patch_hierarchy->getFinestLevelNumber();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      patch_hierarchy->getPatchLevel(ln);

    // We don't need to print this if we are running in serial
    if (n_procs != 1)
      {
        out << "\nrank: " << rank << '\n';
      }
    for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
      {
        bool               printed_value = false;
        std::ostringstream patch_out;
        patch_out << "patch number " << p() << '\n';
        tbox::Pointer<hier::Patch<spacedim>> patch = level->getPatch(p());
        tbox::Pointer<pdat::CellData<spacedim, double>> f_data =
          patch->getPatchData(f_idx);
        const hier::Box<spacedim> patch_box = patch->getBox();

        // elide zero values
        const pdat::ArrayData<spacedim, double> &data = f_data->getArrayData();
        for (pdat::CellIterator<spacedim> i(patch_box); i; i++)
          {
            const int    depth = 0;
            const double value = data(i(), depth);
            if (std::abs(value) > 0)
              {
                patch_out << "array" << i() << " = " << int(value) << '\n';
                printed_value = true;
              }
          }
        if (printed_value)
          out << patch_out.str();
      }
  }

  std::ofstream output;
  if (rank == 0)
    output.open("output");
  print_strings_on_0(out.str(), tbox::SAMRAI_MPI::getCommunicator(), output);
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
