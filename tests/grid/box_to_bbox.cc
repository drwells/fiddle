#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>

#include <deal.II/base/mpi.h>

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

#include "../tests.h"

// NodalPatchMap with multiple levels

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

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);

  // Now set up fiddle things for the test:
  std::ostringstream out;

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
    {
      out << "Patch Level = " << ln << std::endl;
      tbox::Pointer<hier::PatchLevel<spacedim>> level =
        patch_hierarchy->getPatchLevel(ln);
      AssertThrow(level, fdl::ExcFDLInternalError());

      for (int i = 0; i < level->getNumberOfPatches(); ++i)
        {
          tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geom =
            level->getPatch(i)->getPatchGeometry();
          AssertThrow(geom, fdl::ExcInternalError());

          Point<spacedim> lower, upper;
          for (unsigned int d = 0; d < spacedim; ++d)
            {
              lower[d] = geom->getXLower()[d];
              upper[d] = geom->getXUpper()[d];
            }

          out << std::endl;
          out << "Patch Lower = " << lower << std::endl;
          out << "Patch Upper = " << upper << std::endl;

          const auto bbox =
            fdl::box_to_bbox(level->getPatch(i)->getBox(),
                             patch_hierarchy->getPatchLevel(ln));

          out << "Patch bounding box Lower = "
              << bbox.get_boundary_points().first << std::endl;
          out << "Patch bounding box Upper = "
              << bbox.get_boundary_points().second << std::endl;
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
