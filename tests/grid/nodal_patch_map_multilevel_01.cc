#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
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

  // setup deal.II stuff:
  const auto partitioner =
    parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder;
  parallel::shared::Triangulation<dim, spacedim> tria(MPI_COMM_WORLD,
                                                      {},
                                                      false,
                                                      partitioner);
  GridGenerator::hyper_ball(tria);
  tria.refine_global(2);
  {
    std::ofstream out("grid.vtu");
    GridOut().write_vtu(tria, out);
  }

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);

  // setup Lagrangian data:
  const std::size_t n_nodes = tria.n_vertices() * spacedim;
  Vector<double>    nodal_coordinates(n_nodes * spacedim);
  for (std::size_t node_n = 0; node_n < n_nodes; ++node_n)
    for (unsigned int d = 0; d < spacedim; ++d)
      nodal_coordinates[node_n * spacedim + d] = tria.get_vertices()[node_n][d];

  // Now set up fiddle things for the test:
  std::ostringstream                                out;
  std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;
  std::vector<std::vector<BoundingBox<spacedim>>>   bboxes;
  for (int ln = 0; ln <= patch_hierarchy->getFinestLevelNumber(); ++ln)
    {
      const auto new_patches =
        fdl::extract_patches(patch_hierarchy->getPatchLevel(ln));
      patches.insert(patches.end(), new_patches.begin(), new_patches.end());

      if (ln < patch_hierarchy->getFinestLevelNumber())
        {
          const auto nonoverlapping_boxes =
            fdl::compute_nonoverlapping_patch_boxes(
              patch_hierarchy->getPatchLevel(ln),
              patch_hierarchy->getPatchLevel(ln + 1));
          for (const auto &vec : nonoverlapping_boxes)
            {
              bboxes.emplace_back();
              for (const auto &box : vec)
                bboxes.back().push_back(
                  fdl::box_to_bbox(box, patch_hierarchy->getPatchLevel(ln)));
            }
        }
      else
        {
          for (const auto &patch : new_patches)
            {
              bboxes.emplace_back();
              bboxes.back().push_back(
                fdl::box_to_bbox(patch->getBox(),
                                 patch_hierarchy->getPatchLevel(ln)));
            }
        }
    }
  // extend boxes slightly to avoid problems with roundoff - for some reason
  // the center node is not consistently assigned to patches
  for (auto &vec : bboxes)
    for (auto &bbox : vec)
      bbox.extend(1e-6);
  for (const auto &patch : patches)
    {
      tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geom =
        patch->getPatchGeometry();
      AssertThrow(geom, ExcInternalError());

      out << "Patch Level = " << patch->getPatchLevelNumber() << std::endl;
      Point<spacedim> lower, upper;
      for (unsigned int d = 0; d < spacedim; ++d)
        {
          lower[d] = geom->getXLower()[d];
          upper[d] = geom->getXUpper()[d];
        }

      out << "Patch Lower = " << lower << std::endl;
      out << "Patch Upper = " << upper << std::endl;
    }

  fdl::NodalPatchMap<dim, spacedim> nodal_patch_map(patches,
                                                    bboxes,
                                                    nodal_coordinates);

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  {
    const int ln = patch_hierarchy->getFinestLevelNumber();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      patch_hierarchy->getPatchLevel(ln);

    out << std::endl << "NodalPatchMap" << std::endl;
    IndexSet all_indices(nodal_coordinates.size());
    for (std::size_t i = 0; i < nodal_patch_map.size(); ++i)
      {
        const auto &pair = nodal_patch_map[i];
        tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> geom =
          pair.second->getPatchGeometry();
        AssertThrow(geom, ExcInternalError());

        out << "Patch Level = " << pair.second->getPatchLevelNumber()
            << std::endl;

        Point<spacedim> lower, upper;
        for (unsigned int d = 0; d < spacedim; ++d)
          {
            lower[d] = geom->getXLower()[d];
            upper[d] = geom->getXUpper()[d];
          }

        out << "Patch Lower = " << lower << std::endl;
        out << "Patch Upper = " << upper << std::endl;

        out << "indices = ";
        pair.first.print(out);
        out << std::endl;

        const BoundingBox<spacedim> bbox(std::make_pair(lower, upper));
        for (const auto &index : pair.first)
          {
            const Point<spacedim> &vertex =
              tria.get_vertices()[index / spacedim];
            AssertThrow(bbox.point_inside(vertex), fdl::ExcFDLInternalError());
          }

        all_indices.add_indices(pair.first);
      }
    out << "all indices = ";
    all_indices.print(out);
    out << std::endl;
    out << "vector size = " << nodal_coordinates.size() << std::endl;
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
