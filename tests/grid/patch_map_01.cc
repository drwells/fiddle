#include <fiddle/grid/patch_map.h>
#include <fiddle/grid/overlap_tria.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

// Test essential features of PatchMap

int
main(int argc, char **argv)
{
  const auto     mpi_comm = MPI_COMM_WORLD;
  IBTK::IBTKInit ibtk_init(argc, argv, mpi_comm);

  std::ofstream output("output");

  // default constructor
  {
    fdl::PatchMap<2> patch_map;
    output << "empty patch map size = " << patch_map.size() << '\n';
  }

  // Use a patch hierarchy
  {
      using namespace SAMRAI;

      // Input file:
      tbox::Pointer<IBTK::AppInitializer> app_initializer = new IBTK::AppInitializer(argc, argv, "logfile");
      tbox::Pointer<tbox::Database> input_db = app_initializer->getInputDatabase();

      // Set up basic SAMRAI stuff:
      tbox::Pointer<geom::CartesianGridGeometry<2> > grid_geometry
          = new geom::CartesianGridGeometry<2>(
            "CartesianGeometry",
            app_initializer->getComponentDatabase("CartesianGeometry"));
      tbox::Pointer<hier::PatchHierarchy<2> > patch_hierarchy
          = new hier::PatchHierarchy<2>("PatchHierarchy",
                                        grid_geometry);
      tbox::Pointer<mesh::StandardTagAndInitialize<2> > error_detector
          = new mesh::StandardTagAndInitialize<2>(
              "StandardTagAndInitialize", NULL,
              app_initializer->getComponentDatabase("StandardTagAndInitialize"));

      tbox::Pointer<mesh::BergerRigoutsos<2> > box_generator = new mesh::BergerRigoutsos<2>();
      tbox::Pointer<mesh::LoadBalancer<2> > load_balancer =
          new mesh::LoadBalancer<2>("LoadBalancer",
                                    app_initializer->getComponentDatabase("LoadBalancer"));
      tbox::Pointer<mesh::GriddingAlgorithm<2> > gridding_algorithm =
          new mesh::GriddingAlgorithm<2>("GriddingAlgorithm",
                                         app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                         error_detector,
                                         box_generator,
                                         load_balancer);

      // Set up a variable so that we can actually output the grid:
      auto* var_db = hier::VariableDatabase<2>::getDatabase();
      tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");
      tbox::Pointer<pdat::CellVariable<2, double> > u_cc_var
        = new pdat::CellVariable<2, double>("u_cc");
      const int u_cc_idx = var_db->registerVariableAndContext(u_cc_var, ctx,
                                                              hier::IntVector<2>(1));

      gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
      const int tag_buffer = std::numeric_limits<int>::max();
      int level_number = 0;
      while ((gridding_algorithm->levelCanBeRefined(level_number)))
      {
          gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, tag_buffer);
          ++level_number;
      }
      const int finest_level = patch_hierarchy->getFinestLevelNumber();
      for (int ln = 0; ln <= finest_level; ++ln)
      {
          tbox::Pointer<hier::PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(ln);
          level->allocatePatchData(u_cc_idx, 0.0);
      }

      auto visit_data_writer = app_initializer->getVisItDataWriter();
      TBOX_ASSERT(visit_data_writer);
      visit_data_writer->registerPlotQuantity(u_cc_var->getName(), "SCALAR", u_cc_idx);

      // Output SAMRAI plotting information:
      visit_data_writer->writePlotData(patch_hierarchy, 0, 0.0);

      std::vector<tbox::Pointer<hier::Patch<NDIM>>> patches;
      tbox::Pointer<hier::PatchLevel<2> > level = patch_hierarchy->getPatchLevel(finest_level);
      int local_patch_num = 0;
      for (hier::PatchLevel<2>::Iterator p(level); p; p++)
        patches.emplace_back(level->getPatch(p()));

      // Set up deal.II and fiddle stuff
      {
        using namespace dealii;

        Triangulation<2> tria;
        GridGenerator::hyper_ball(tria);
        tria.refine_global(2);

        std::vector<BoundingBox<2>> cell_bboxes;
        for (const auto cell : tria.active_cell_iterators())
          cell_bboxes.push_back(cell->bounding_box());

        // Set up the relevant fiddle class:
        fdl::PatchMap<2> patch_map(patches, 1.0, tria, cell_bboxes);

        // now do the actual test
        FE_Nothing<2> fe;
        DoFHandler<2> dof_handler(tria);
        dof_handler.distribute_dofs(fe);

        for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
        {
          std::vector<BoundingBox<2>> cell_patch_bboxes;
          auto iterator = patch_map.begin(patch_n, dof_handler);
          const auto end = patch_map.end(patch_n, dof_handler);
          for (; iterator != end; ++iterator)
            {
              cell_patch_bboxes.emplace_back((*iterator)->bounding_box());
            }

          output << "Number of FE cells on patch "
                 << patch_n
                 << " = "
                 << cell_patch_bboxes.size() << '\n';

          BoundingBoxDataOut<2> bbox_data_out;
          bbox_data_out.build_patches(cell_patch_bboxes);
          std::ofstream viz_out("out-" + std::to_string(patch_n) + ".vtk");

          DataOutBase::VtkFlags flags;
          flags.print_date_and_time = false;
          bbox_data_out.set_flags(flags);
          bbox_data_out.write_vtk(viz_out);

          output << "bboxes on patch " << patch_n << '\n';
          bbox_data_out.write_vtk(output);
        }
      }
  }
}
