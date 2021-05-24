#include <fiddle/interaction/ifed_method.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/BoxPartitioner.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <fstream>
#include <vector>

#include "../tests.h"

// Test tagging when combined with all the IBAMR stuff

using namespace SAMRAI;
using namespace dealii;

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD);
  Point<dim> center;
  center[0] = 0.6;
  center[1] = 0.5;
  center[dim - 1] = 0.5; // works in 2D and 3D
  GridGenerator::hyper_ball(native_tria, center, 0.2);
  native_tria.refine_global(std::log2(input_db->getInteger("N")));

  // fiddle stuff:
  FESystem<dim> fe(FE_Q<dim>(1), dim);
  std::vector<fdl::Part<dim>> parts;
  parts.emplace_back(native_tria, fe);
  tbox::Pointer<IBAMR::IBStrategy> ib_method_ops = new fdl::IFEDMethod<dim>(std::move(parts));

  // Create major algorithm and data objects that comprise the
  // application.  These objects are configured from the input database
  // and, if this is a restarted run, from the restart database.
  tbox::Pointer<geom::CartesianGridGeometry<spacedim> > grid_geometry = new geom::CartesianGridGeometry<spacedim>(
      "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
  tbox::Pointer<hier::PatchHierarchy<spacedim> > patch_hierarchy = new hier::PatchHierarchy<spacedim>("PatchHierarchy", grid_geometry);
  tbox::Pointer<mesh::LoadBalancer<spacedim> > load_balancer =
      new mesh::LoadBalancer<spacedim>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
  tbox::Pointer<mesh::BergerRigoutsos<spacedim> > box_generator = new mesh::BergerRigoutsos<spacedim>();

  tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator
      = new IBAMR::INSStaggeredHierarchyIntegrator(
          "INSStaggeredHierarchyIntegrator",
          app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));

  tbox::Pointer<IBAMR::IBHierarchyIntegrator> time_integrator =
      new IBAMR::IBExplicitHierarchyIntegrator(
          "IBHierarchyIntegrator",
          app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
          ib_method_ops,
          navier_stokes_integrator);
  time_integrator->registerLoadBalancer(load_balancer);

  tbox::Pointer<mesh::StandardTagAndInitialize<spacedim> > error_detector =
      new mesh::StandardTagAndInitialize<spacedim>(
          "StandardTagAndInitialize",
          time_integrator,
          app_initializer->getComponentDatabase("StandardTagAndInitialize"));
  tbox::Pointer<mesh::GriddingAlgorithm<spacedim> > gridding_algorithm =
      new mesh::GriddingAlgorithm<spacedim>(
          "GriddingAlgorithm",
          app_initializer->getComponentDatabase("GriddingAlgorithm"),
          error_detector,
          box_generator,
          load_balancer);

  std::vector<solv::RobinBcCoefStrategy<spacedim>*> u_bc_coefs(spacedim);
  // Create Eulerian boundary condition specification objects.
  for (int d = 0; d < spacedim; ++d)
  {
      const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

      const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

      u_bc_coefs[d] = new IBTK::muParserRobinBcCoefs(
          bc_coefs_name, app_initializer->getComponentDatabase(bc_coefs_db_name), grid_geometry);
  }
  navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);

  // Set up visualization plot file writers.
  tbox::Pointer<appu::VisItDataWriter<spacedim> > visit_data_writer = app_initializer->getVisItDataWriter();
  time_integrator->registerVisItDataWriter(visit_data_writer);

  // Initialize hierarchy configuration and data on all patches.
  time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

  // Write out initial visualization data.
  int iteration_num = time_integrator->getIntegratorStep();
  double loop_time = time_integrator->getIntegratorTime();
  time_integrator->setupPlotData();
  visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);

  // Main time step loop.
  double loop_time_end = time_integrator->getEndTime();
  double dt = 0.0;
  while (!tbox::MathUtilities<double>::equalEps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
  {
      iteration_num = time_integrator->getIntegratorStep();
      loop_time = time_integrator->getIntegratorTime();

      tbox::pout << "\n";
      tbox::pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
      tbox::pout << "At beginning of timestep # " << iteration_num << "\n";
      tbox::pout << "Simulation time is " << loop_time << "\n";

      dt = time_integrator->getMaximumTimeStepSize();
      time_integrator->advanceHierarchy(dt);
      loop_time += dt;

      tbox::pout << "\n";
      tbox::pout << "At end       of timestep # " << iteration_num << "\n";
      tbox::pout << "Simulation time is " << loop_time << "\n";
      tbox::pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
      tbox::pout << "\n";

      // At specified intervals, write visualization and restart files,
      // print out timer data, and store hierarchy data for post
      // processing.
      iteration_num += 1;
      const bool last_step = !time_integrator->stepsRemaining();
      if (last_step || iteration_num % 10 == 0)
      {
          tbox::pout << "\nWriting visualization files...\n\n";
          time_integrator->setupPlotData();
          visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
      }
  }

  // Actual test: print out the boxes at the end. We didn't do enough timesteps
  // to regrid so this should be consistent (and independent of velocity
  // interpolation).
  {
      std::ofstream out("output");
      print_partitioning_on_0(patch_hierarchy, 0, patch_hierarchy->getFinestLevelNumber(), out);
  }

  for (auto ptr : u_bc_coefs)
      delete ptr;
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "ifed_tag.log");

  test<NDIM>(app_initializer);
}
