#include <fiddle/base/exceptions.h>
#include <fiddle/base/samrai_utilities.h>

#include <deal.II/base/exceptions.h>

#include <ibtk/CartSideDoubleSpecializedLinearRefine.h>
#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/muParserCartGridFunction.h>

#include <BergerRigoutsos.h>
#include <Box.h>
#include <CartesianGridGeometry.h>
#include <GriddingAlgorithm.h>
#include <LoadBalancer.h>
#include <PatchHierarchy.h>
#include <StandardTagAndInitialize.h>
#include <tbox/SAMRAI_MPI.h>

#include <mpi.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

// utility function for getting the number of components in the test subdatabase
// of the input database.
int
get_n_f_components(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db)
{
  auto test_db        = input_db->getDatabase("test");
  int  n_f_components = 0;
  if (test_db->keyExists("f"))
    n_f_components = test_db->getDatabase("f")->getAllKeys().getSize();
  else
    n_f_components = test_db->getIntegerWithDefault("n_components", 1);

  return n_f_components;
}

std::string
extract_fp_string(SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> f_db)
{
  std::string fp_string;
  const int   n_F_components = f_db->getAllKeys().getSize();
  if (n_F_components == 1)
    {
      fp_string += f_db->getString("function");
    }
  else
    {
      for (int c = 0; c < n_F_components; ++c)
        {
          fp_string += f_db->getString("function_" + std::to_string(c));
          if (c != n_F_components - 1)
            fp_string += ';';
        }
    }
  return fp_string;
}

// A utility function that does the normal SAMRAI initialization stuff.
template <int spacedim>
std::pair<SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<spacedim>>, int>
setup_hierarchy(
  SAMRAI::tbox::Pointer<IBTK::AppInitializer>         app_initializer,
  SAMRAI::mesh::StandardTagAndInitStrategy<spacedim> *tag = nullptr)
{
  using namespace SAMRAI;

  auto input_db = app_initializer->getInputDatabase();

  // database specific to tests, should it exist
  auto test_db = input_db;
  if (input_db->keyExists("test"))
    test_db = input_db->getDatabase("test");

  // Set up basic SAMRAI stuff:
  tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geometry =
    new geom::CartesianGridGeometry<spacedim>(
      "CartesianGeometry", input_db->getDatabase("CartesianGeometry"));

  // More questionable SAMRAI design decisions - we have to register refine
  // operations associated with different variables with the grid geometry class
  grid_geometry->addSpatialRefineOperator(
    new geom::CartesianCellDoubleLinearRefine<spacedim>());
  grid_geometry->addSpatialRefineOperator(
    new IBTK::CartSideDoubleSpecializedLinearRefine());

  tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy =
    new hier::PatchHierarchy<spacedim>("PatchHierarchy", grid_geometry);
  tbox::Pointer<mesh::StandardTagAndInitialize<spacedim>> error_detector =
    new mesh::StandardTagAndInitialize<spacedim>("StandardTagAndInitialize",
                                                 tag,
                                                 input_db->getDatabase(
                                                   "StandardTagAndInitialize"));

  tbox::Pointer<mesh::BergerRigoutsos<spacedim>> box_generator =
    new mesh::BergerRigoutsos<spacedim>();
  tbox::Pointer<mesh::LoadBalancer<spacedim>> load_balancer =
    new mesh::LoadBalancer<spacedim>("LoadBalancer",
                                     input_db->getDatabase("LoadBalancer"));
  tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_algorithm =
    new mesh::GriddingAlgorithm<spacedim>("GriddingAlgorithm",
                                          input_db->getDatabase(
                                            "GriddingAlgorithm"),
                                          error_detector,
                                          box_generator,
                                          load_balancer);

  // Set up a variable so that we can actually output the grid. Note that this
  // has to happen before we make any levels since this is where we set the
  // maximum ghost width.
  auto *var_db = hier::VariableDatabase<spacedim>::getDatabase();
  tbox::Pointer<hier::VariableContext> ctx = var_db->getContext("context");

  const int n_f_components = get_n_f_components(input_db);

  tbox::Pointer<hier::Variable<spacedim>> f_var;
  const std::string                       f_data_type =
    test_db->getStringWithDefault("f_data_type", "CELL");
  if (f_data_type == "CELL")
    f_var = new pdat::CellVariable<spacedim, double>("f_cc", n_f_components);
  else if (f_data_type == "SIDE")
    // This one is different since side-centered data already has implicitly
    // spacedim 'depth' in a different sense
    //
    // TODO this should probably be clarified somehow
    {
      AssertThrow(n_f_components == spacedim,
                  dealii::ExcMessage("The only supported number of components "
                                     "for side-centered data is the spatial "
                                     "dimension"));
      f_var = new pdat::SideVariable<spacedim, double>("f_sc", 1);
    }
  else if (f_data_type == "NODE")
    f_var = new pdat::SideVariable<spacedim, double>("f_nc", n_f_components);
  else
    AssertThrow(false, fdl::ExcFDLNotImplemented());

  // BSPLINE-3 needs 3 ghost cells
  const int ghost_width = test_db->getIntegerWithDefault("ghost_width", 3);
  const int f_idx =
    var_db->registerVariableAndContext(f_var,
                                       ctx,
                                       hier::IntVector<spacedim>(ghost_width));

  // set up the SAMRAI grid:
  gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
  int       level_number = 0;
  const int tag_buffer   = 1;
  while (gridding_algorithm->levelCanBeRefined(level_number))
    {
      gridding_algorithm->makeFinerLevel(patch_hierarchy, 0.0, 0.0, tag_buffer);
      ++level_number;
    }

  const int finest_level = patch_hierarchy->getFinestLevelNumber();
  for (int ln = 0; ln <= finest_level; ++ln)
    {
      tbox::Pointer<hier::PatchLevel<spacedim>> level =
        patch_hierarchy->getPatchLevel(ln);
      level->allocatePatchData(f_idx, 0.0);

      auto patches = fdl::extract_patches(level);
      for (auto &patch : patches)
        fdl::fill_all(patch->getPatchData(f_idx), 42);
    }

  auto visit_data_writer = app_initializer->getVisItDataWriter();
  TBOX_ASSERT(visit_data_writer);
  // Yet another SAMRAI bug - non-cell data cannot be plotted
  int                                     plot_cc_idx = 0;
  tbox::Pointer<hier::Variable<spacedim>> plot_cc_var;
  if (f_data_type == "CELL")
    for (unsigned int d = 0; d < n_f_components; ++d)
      visit_data_writer->registerPlotQuantity(
        f_var->getName() + std::to_string(d), "SCALAR", f_idx, d);
  else
    {
      plot_cc_var =
        new pdat::CellVariable<spacedim, double>("f_cc", n_f_components);
      plot_cc_idx =
        var_db->registerVariableAndContext(plot_cc_var,
                                           ctx,
                                           hier::IntVector<spacedim>(0));

      for (int ln = 0; ln <= finest_level; ++ln)
        {
          tbox::Pointer<hier::PatchLevel<spacedim>> level =
            patch_hierarchy->getPatchLevel(ln);
          level->allocatePatchData(plot_cc_idx, 0.0);

          auto patches = fdl::extract_patches(level);
          for (auto &patch : patches)
            fdl::fill_all(patch->getPatchData(plot_cc_idx), 0);
        }

      for (unsigned int d = 0; d < n_f_components; ++d)
        {
          visit_data_writer->registerPlotQuantity(plot_cc_var->getName() +
                                                    std::to_string(d),
                                                  "SCALAR",
                                                  plot_cc_idx,
                                                  d);
        }
    }

  // If it exists, set up an initial condition
  if (test_db->keyExists("f"))
    {
      IBTK::muParserCartGridFunction f_fcn("f",
                                           test_db->getDatabase("f"),
                                           patch_hierarchy->getGridGeometry());
      f_fcn.setDataOnPatchHierarchy(f_idx, f_var, patch_hierarchy, 0.0);

      using ITC = IBTK::HierarchyGhostCellInterpolation::
        InterpolationTransactionComponent;
      std::vector<ITC> ghost_cell_components(1);
      // TODO - the way to select ghost filling algorithms is a horrible mess
      const std::string refine_type =
        f_data_type == "CELL" ? "LINEAR_REFINE" : "SPECIALIZED_LINEAR_REFINE";
      ghost_cell_components[0] = ITC(f_idx,
                                     refine_type,
                                     true,
                                     "CONSERVATIVE_COARSEN",
                                     "LINEAR",
                                     false,
                                     {}, // f_bc_coefs
                                     nullptr);
      IBTK::HierarchyGhostCellInterpolation ghost_fill_op;
      ghost_fill_op.initializeOperatorState(ghost_cell_components,
                                            patch_hierarchy);
      ghost_fill_op.fillData(/*time*/ 0.0);
    }

  return std::make_pair(patch_hierarchy, f_idx);
}

// A utility function that prints @p part_str to @p out by sending each string to
// rank 0.
inline void
print_strings_on_0(const std::string &part_str, std::ofstream &out)
{
  using namespace SAMRAI::tbox;
  const int                  n_nodes = SAMRAI_MPI::getNodes();
  std::vector<unsigned long> string_sizes(n_nodes);

  const unsigned long size = part_str.size();

  int ierr = MPI_Gather(&size,
                        1,
                        MPI_UNSIGNED_LONG,
                        string_sizes.data(),
                        1,
                        MPI_UNSIGNED_LONG,
                        0,
                        SAMRAI_MPI::getCommunicator());
  TBOX_ASSERT(ierr == 0);

  // MPI_Gatherv would be more efficient, but this just a test so its
  // not too important
  if (SAMRAI_MPI::getRank() == 0)
    {
      out << part_str;
      for (int r = 1; r < n_nodes; ++r)
        {
          std::string input;
          input.resize(string_sizes[r]);
          ierr = MPI_Recv(&input[0],
                          string_sizes[r],
                          MPI_CHAR,
                          r,
                          0,
                          SAMRAI_MPI::getCommunicator(),
                          MPI_STATUS_IGNORE);
          TBOX_ASSERT(ierr == 0);
          out << input;
        }
    }
  else
    MPI_Send(
      part_str.data(), size, MPI_CHAR, 0, 0, SAMRAI_MPI::getCommunicator());
}

/**
 * Print the parallel partitioning (i.e., the boxes) on all processes to @p out
 * on processor 0.
 */
template <int spacedim>
inline void
print_partitioning_on_0(
  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<spacedim>>
    &            patch_hierarchy,
  const int      coarsest_ln,
  const int      finest_ln,
  std::ofstream &out)
{
  using namespace SAMRAI;

  std::ostringstream part_steam;
  for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
      tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
        patch_hierarchy->getPatchLevel(ln);
      part_steam << "rank: " << tbox::SAMRAI_MPI::getRank() << " level: " << ln
                 << " boxes:\n";
      for (typename hier::PatchLevel<spacedim>::Iterator p(patch_level); p; p++)
        {
          const hier::Box<spacedim> box = patch_level->getPatch(p())->getBox();
          part_steam << box << '\n';
        }
    }
  print_strings_on_0(part_steam.str(), out);
}
