#include <deal.II/base/exceptions.h>

#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/muParserCartGridFunction.h>

#include <Box.h>
#include <PatchHierarchy.h>
#include <tbox/SAMRAI_MPI.h>

#include <mpi.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

// A utility function that does the normal SAMRAI initialization stuff.
template <int spacedim>
std::pair<SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<spacedim>>,
          int>
setup_hierarchy(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer,
                SAMRAI::mesh::StandardTagAndInitStrategy<spacedim> *tag = nullptr)
{
  using namespace SAMRAI;

  auto input_db = app_initializer->getInputDatabase();

  // Set up basic SAMRAI stuff:
  tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geometry =
    new geom::CartesianGridGeometry<spacedim>(
      "CartesianGeometry", input_db->getDatabase("CartesianGeometry"));

  // TODO: do we need this at all?
  tbox::Pointer<xfer::RefineOperator<spacedim> > linear_refine =
    new geom::CartesianCellDoubleLinearRefine<spacedim>();
  grid_geometry->addSpatialRefineOperator(linear_refine);

  tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy =
    new hier::PatchHierarchy<spacedim>("PatchHierarchy", grid_geometry);
  tbox::Pointer<mesh::StandardTagAndInitialize<spacedim>> error_detector =
    new mesh::StandardTagAndInitialize<spacedim>(
      "StandardTagAndInitialize",
      tag,
      input_db->getDatabase("StandardTagAndInitialize"));

  tbox::Pointer<mesh::BergerRigoutsos<spacedim>> box_generator =
    new mesh::BergerRigoutsos<spacedim>();
  tbox::Pointer<mesh::LoadBalancer<spacedim>> load_balancer =
    new mesh::LoadBalancer<spacedim>(
      "LoadBalancer", input_db->getDatabase("LoadBalancer"));
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

  const int n_F_components = input_db->getIntegerWithDefault("n_components", 1);
  tbox::Pointer<pdat::CellVariable<spacedim, double>> f_cc_var =
    new pdat::CellVariable<spacedim, double>(
      "f_cc", n_F_components);
  const int f_cc_idx =
    var_db->registerVariableAndContext(f_cc_var,
                                       ctx,
                                       // need 3 for bspline 3 ghosts
                                       hier::IntVector<spacedim>(3));

  // set up the SAMRAI grid:
  gridding_algorithm->makeCoarsestLevel(patch_hierarchy, 0.0);
  int level_number = 0;
  const int tag_buffer = 1;
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
      level->allocatePatchData(f_cc_idx, 0.0);

      // obviously this won't generalize well
      auto patches = fdl::extract_patches(level);
      for (auto &patch : patches)
        {
          tbox::Pointer<pdat::CellData<spacedim, double>> data =
            patch->getPatchData(f_cc_idx);
          Assert(data, dealii::ExcMessage("pointer should not be null"));
          data->fillAll(42.0);
        }
    }

  auto visit_data_writer = app_initializer->getVisItDataWriter();
  TBOX_ASSERT(visit_data_writer);
  for (unsigned int d = 0; d < n_F_components; ++d)
    {
      visit_data_writer->registerPlotQuantity(f_cc_var->getName() + std::to_string(d), "SCALAR", f_cc_idx, d);
    }

  // If it exists, set up an initial condition
  if (input_db->keyExists("f"))
    {
      IBTK::muParserCartGridFunction f_fcn(
        "f",
        input_db->getDatabase("f"),
        patch_hierarchy->getGridGeometry());
      auto* var_db = hier::VariableDatabase<spacedim>::getDatabase();
      tbox::Pointer<hier::Variable<spacedim>> var;
      var_db->mapIndexToVariable(f_cc_idx, var);
      f_fcn.setDataOnPatchHierarchy(f_cc_idx,
                                    var,
                                    patch_hierarchy,
                                    0.0);

      using ITC = IBTK::HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
      std::vector<ITC> ghost_cell_components(1);
      ghost_cell_components[0] = ITC(f_cc_idx,
                                     "LINEAR_REFINE",
                                     true,
                                     "CONSERVATIVE_COARSEN",
                                     "LINEAR",
                                     false,
                                     {}, // f_bc_coefs
                                     nullptr);
      IBTK::HierarchyGhostCellInterpolation ghost_fill_op;
      ghost_fill_op.initializeOperatorState(ghost_cell_components, patch_hierarchy);
      ghost_fill_op.fillData(/*time*/ 0.0);
    }

  return std::make_pair(patch_hierarchy, f_cc_idx);
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
