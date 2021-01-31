#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/rtree.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>

// Test essential features of PatchMap + OverlappingTriangulation

int
main(int argc, char **argv)
{
  const auto     mpi_comm = MPI_COMM_WORLD;
  IBTK::IBTKInit ibtk_init(argc, argv, mpi_comm);

  const auto    rank    = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
  const auto    n_procs = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
  std::ofstream output("output-" + std::to_string(rank));

  // Use a patch hierarchy
  {
    using namespace SAMRAI;

    // Input file:
    tbox::Pointer<IBTK::AppInitializer> app_initializer =
      new IBTK::AppInitializer(argc, argv, "logfile");
    tbox::Pointer<tbox::Database> input_db =
      app_initializer->getInputDatabase();

    // Set up basic SAMRAI stuff:
    tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
      new geom::CartesianGridGeometry<NDIM>(
        "CartesianGeometry",
        app_initializer->getComponentDatabase("CartesianGeometry"));
    tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy =
      new hier::PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
    tbox::Pointer<mesh::StandardTagAndInitialize<NDIM>> error_detector =
      new mesh::StandardTagAndInitialize<NDIM>(
        "StandardTagAndInitialize",
        NULL,
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

    // Set up a variable so that we can actually output the grid:
    auto *var_db = hier::VariableDatabase<NDIM>::getDatabase();
    tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");
    tbox::Pointer<pdat::CellVariable<2, double>> u_cc_var =
      new pdat::CellVariable<2, double>("u_cc");
    const int u_cc_idx =
      var_db->registerVariableAndContext(u_cc_var,
                                         ctx,
                                         hier::IntVector<NDIM>(1));

    gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
    const int tag_buffer   = std::numeric_limits<int>::max();
    int       level_number = 0;
    while ((gridding_algorithm->levelCanBeRefined(level_number)))
      {
        gridding_algorithm->makeFinerLevel(patch_hierarchy,
                                           0.0,
                                           0.0,
                                           tag_buffer);
        ++level_number;
      }
    const int finest_level = patch_hierarchy->getFinestLevelNumber();
    for (int ln = 0; ln <= finest_level; ++ln)
      {
        tbox::Pointer<hier::PatchLevel<NDIM>> level =
          patch_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(u_cc_idx, 0.0);
      }

    auto visit_data_writer = app_initializer->getVisItDataWriter();
    TBOX_ASSERT(visit_data_writer);
    visit_data_writer->registerPlotQuantity(u_cc_var->getName(),
                                            "SCALAR",
                                            u_cc_idx);

    // Output SAMRAI plotting information:
    visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);

    std::vector<tbox::Pointer<hier::Patch<NDIM>>> patches;
    tbox::Pointer<hier::PatchLevel<NDIM>>         level =
      patch_hierarchy->getPatchLevel(finest_level);
    int local_patch_num = 0;
    for (hier::PatchLevel<NDIM>::Iterator p(level); p; p++)
      patches.emplace_back(level->getPatch(p()));

    // Set up deal.II and fiddle stuff
    {
      using namespace dealii;

      parallel::shared::Triangulation<NDIM> native_tria(mpi_comm);
      GridGenerator::hyper_ball(native_tria);
      native_tria.refine_global(6 - NDIM);

      const std::vector<BoundingBox<NDIM>> patch_bboxes =
        fdl::compute_patch_bboxes(patches, 1.0);
      fdl::TriaIntersectionPredicate<NDIM> tria_pred(patch_bboxes);
      fdl::OverlapTriangulation<NDIM>      overlap_tria(native_tria, tria_pred);

      std::vector<BoundingBox<NDIM>> cell_bboxes;
      for (const auto cell : overlap_tria.active_cell_iterators())
        cell_bboxes.push_back(cell->bounding_box());

      // Set up the relevant fiddle class:
      fdl::PatchMap<NDIM> patch_map(patches, 1.0, overlap_tria, cell_bboxes);

      // now do the actual test
      FE_Nothing<NDIM> fe;
      DoFHandler<NDIM> dof_handler(overlap_tria);
      dof_handler.distribute_dofs(fe);

      for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
        {
          std::vector<BoundingBox<NDIM>> cell_patch_bboxes;
          auto       iterator = patch_map.begin(patch_n, dof_handler);
          const auto end      = patch_map.end(patch_n, dof_handler);
          for (; iterator != end; ++iterator)
            {
              cell_patch_bboxes.emplace_back((*iterator)->bounding_box());
            }

          output << "Number of FE cells on patch " << patch_n << " = "
                 << cell_patch_bboxes.size() << '\n';

          GridOut       go;
          std::ofstream grid_out("overlap-" + std::to_string(rank) + ".vtk");
          go.write_vtk(overlap_tria, grid_out);

          // print out the RTree
          {
            auto rtree      = pack_rtree(cell_patch_bboxes);
            namespace bgi   = boost::geometry::index;
            using RTree     = std::remove_const<decltype(rtree)>::type;
            using RtreeView = bgi::detail::rtree::utilities::view<RTree>;
            RtreeView rtv(rtree);

            for (unsigned int level_n = 0; level_n < rtv.depth(); ++level_n)
              {
                const auto level_boxes = extract_rtree_level(rtree, level_n);

                BoundingBoxDataOut<NDIM> bbox_data_out_level;
                bbox_data_out_level.build_patches(level_boxes);
                std::ofstream viz_out("rtree-r-p-l-" + std::to_string(rank) +
                                      "-" + std::to_string(patch_n) + "-" +
                                      std::to_string(level_n) + ".vtk");

                DataOutBase::VtkFlags flags;
                flags.print_date_and_time = false;
                bbox_data_out_level.set_flags(flags);
                bbox_data_out_level.write_vtk(viz_out);
                bbox_data_out_level.write_vtk(output);
              }
          }
          output << "OK\n";
        }
    }
  }

  MPI_Barrier(mpi_comm);

  if (rank == 0)
    {
      std::ofstream output("output");
      for (unsigned int r = 0; r < n_procs; ++r)
        {
          output << "===================== "
                 << "output on rank " << r << " ====================="
                 << "\n\n";
          std::ifstream in("output-" + std::to_string(r));
          output << in.rdbuf() << "\n";
        }
    }
}
