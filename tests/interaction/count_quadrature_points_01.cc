#include <fiddle/base/exceptions.h>

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

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

// Test count_quadrature_points

using namespace SAMRAI;
using namespace dealii;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);
  const auto n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_ball(native_tria);
  // Even though we are periodic in both directions we don't ever need to
  // actually enforce this in the finite element code as far as spreading goes
  native_tria.refine_global(std::log2(input_db->getInteger("N")/2));

  // setup SAMRAI stuff (its always the same):
  auto pair            = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = pair.first;
  auto f_idx           = pair.second;

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
  for (const auto cell : overlap_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      cell_bboxes.push_back(fbbox);
    }
  fdl::PatchMap<dim, spacedim> patch_map(patches,
                                         1.0,
                                         overlap_tria,
                                         cell_bboxes);

  // set up what we need to count quadrature points:
  const MappingQ<dim>                X_map(2);
  const std::vector<Quadrature<dim>> quadratures({QMidpoint<dim>()});
  const std::vector<unsigned char>   quadrature_indices(
    overlap_tria.n_active_cells());

  fdl::count_quadrature_points(f_idx,
                               patch_map,
                               X_map,
                               quadrature_indices,
                               quadratures);

  {
    std::ofstream out("output-" + std::to_string(rank));
    GridOut       go;
    go.write_vtk(overlap_tria, out);
  }

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
    tbox::Pointer<hier::PatchLevel<spacedim> > level = patch_hierarchy->getPatchLevel(ln);

    // We don't need to print this if we are running in serial
    if (n_procs != 1)
      {
        out << "\nrank: " << rank << '\n';
      }
    for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
      {
        bool printed_value = false;
        std::ostringstream patch_out;
        patch_out << "patch number " << p() << '\n';
        tbox::Pointer<hier::Patch<spacedim> > patch = level->getPatch(p());
        tbox::Pointer<pdat::CellData<spacedim, double> > f_data = patch->getPatchData(f_idx);
        const hier::Box<spacedim> patch_box = patch->getBox();

        // elide zero values
        const pdat::ArrayData<spacedim, double>& data = f_data->getArrayData();
        for (pdat::CellIterator<spacedim> i(patch_box); i; i++)
        {
          const int depth = 0;
          const double value = data(i(), depth);
          if (std::abs(value) > 0)
            {
              patch_out << "array" << i() << " = " << int(value) << '\n';
              printed_value = true;
            }
        }
        if (printed_value) out << patch_out.str();
      }
  }

  std::ofstream output;
  if (rank == 0)
    output.open("output");
  print_strings_on_0(out.str(), output);

}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
