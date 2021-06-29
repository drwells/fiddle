#include <fiddle/interaction/ifed_method.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>
#include <vector>

#include "../tests.h"

// Test interpolation

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto       input_db = app_initializer->getInputDatabase();
  auto       test_db  = input_db->getDatabase("test");
  const auto mpi_comm = MPI_COMM_WORLD;

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(
    mpi_comm, {}, test_db->getBoolWithDefault("use_artificial_cells", false));

  Point<dim> center;
  center[0]       = 0.6;
  center[1]       = 0.5;
  center[dim - 1] = 0.5; // works in 2D and 3D
  GridGenerator::hyper_ball(native_tria, center, 0.2);
  native_tria.refine_global(3);

  tbox::pout << "Number of elements = " << native_tria.n_active_cells() << '\n';

  // fiddle stuff:
  FESystem<dim>               fe(FE_Q<dim>(1), dim);
  std::vector<fdl::Part<dim>> parts;
  parts.emplace_back(native_tria, fe);
  tbox::Pointer<IBAMR::IBStrategy> ib_method_ops =
    new fdl::IFEDMethod<dim>(input_db->getDatabase("IFEDMethod"),
                             std::move(parts));

  // Create major algorithm and data objects that comprise the
  // application.  These objects are configured from the input database
  // and, if this is a restarted run, from the restart database.
  tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geometry =
    new geom::CartesianGridGeometry<spacedim>(
      "CartesianGeometry",
      app_initializer->getComponentDatabase("CartesianGeometry"));
  tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy =
    new hier::PatchHierarchy<spacedim>("PatchHierarchy", grid_geometry);
  tbox::Pointer<mesh::LoadBalancer<spacedim>> load_balancer =
    new mesh::LoadBalancer<spacedim>(
      "LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
  tbox::Pointer<mesh::BergerRigoutsos<spacedim>> box_generator =
    new mesh::BergerRigoutsos<spacedim>();

  tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator =
    new IBAMR::INSStaggeredHierarchyIntegrator(
      "INSStaggeredHierarchyIntegrator",
      app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));

  tbox::Pointer<IBAMR::IBHierarchyIntegrator> time_integrator =
    new IBAMR::IBExplicitHierarchyIntegrator(
      "IBHierarchyIntegrator",
      app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
      ib_method_ops,
      navier_stokes_integrator);
  time_integrator->registerLoadBalancer(load_balancer);

  tbox::Pointer<mesh::StandardTagAndInitialize<spacedim>> error_detector =
    new mesh::StandardTagAndInitialize<spacedim>(
      "StandardTagAndInitialize",
      time_integrator,
      app_initializer->getComponentDatabase("StandardTagAndInitialize"));
  tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_algorithm =
    new mesh::GriddingAlgorithm<spacedim>("GriddingAlgorithm",
                                          app_initializer->getComponentDatabase(
                                            "GriddingAlgorithm"),
                                          error_detector,
                                          box_generator,
                                          load_balancer);

  std::vector<solv::RobinBcCoefStrategy<spacedim> *> u_bc_coefs(spacedim);
  // Create Eulerian boundary condition specification objects.
  for (int d = 0; d < spacedim; ++d)
    {
      const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

      const std::string bc_coefs_db_name =
        "VelocityBcCoefs_" + std::to_string(d);

      u_bc_coefs[d] =
        new IBTK::muParserRobinBcCoefs(bc_coefs_name,
                                       app_initializer->getComponentDatabase(
                                         bc_coefs_db_name),
                                       grid_geometry);
    }
  navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);

  // Set up visualization plot file writers.
  tbox::Pointer<appu::VisItDataWriter<spacedim>> visit_data_writer =
    app_initializer->getVisItDataWriter();
  time_integrator->registerVisItDataWriter(visit_data_writer);

  // Initialize hierarchy configuration and data on all patches.
  time_integrator->initializePatchHierarchy(patch_hierarchy,
                                            gridding_algorithm);

  // Write out initial visualization data.
  int    iteration_num = time_integrator->getIntegratorStep();
  double loop_time     = time_integrator->getIntegratorTime();
  time_integrator->setupPlotData();
  visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);

  // Main time step loop.
  double loop_time_end = time_integrator->getEndTime();
  double dt            = 0.0;
  while (!tbox::MathUtilities<double>::equalEps(loop_time, loop_time_end) &&
         time_integrator->stepsRemaining())
    {
      iteration_num = time_integrator->getIntegratorStep();
      loop_time     = time_integrator->getIntegratorTime();

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

      iteration_num += 1;
      const bool last_step = !time_integrator->stepsRemaining();
      if (last_step || iteration_num % 10 == 0)
        {
          tbox::pout << "\nWriting visualization files...\n\n";
          time_integrator->setupPlotData();
          visit_data_writer->writePlotData(patch_hierarchy,
                                           iteration_num,
                                           loop_time);

          const auto &part =
            dynamic_cast<fdl::IFEDMethod<NDIM> &>(*ib_method_ops).get_part(0);
          DataOut<dim> data_out;
          data_out.attach_dof_handler(part.get_dof_handler());
          data_out.add_data_vector(part.get_velocity(), "U");

          MappingFEField<dim,
                         spacedim,
                         LinearAlgebra::distributed::Vector<double>>
            X_mapping(part.get_dof_handler(), part.get_position());
          data_out.build_patches(X_mapping);
          data_out.write_vtu_with_pvtu_record(
            "./", "solution", iteration_num, mpi_comm, 8);
        }
    }

  // Save approximate cell centers as the test output
  {
    const auto         rank = Utilities::MPI::this_mpi_process(mpi_comm);
    std::ostringstream proc_out;
    proc_out << "rank = " << rank << '\n';
    proc_out << std::setprecision(10);

    const auto &part =
      dynamic_cast<fdl::IFEDMethod<NDIM> &>(*ib_method_ops).get_part(0);

    MappingFEField<dim, spacedim, LinearAlgebra::distributed::Vector<double>>
      X_mapping(part.get_dof_handler(), part.get_position());

    QMidpoint<dim> quad;
    FEValues<dim>  fe_values(X_mapping,
                            part.get_dof_handler().get_fe(),
                            quad,
                            update_quadrature_points | update_values);

    std::vector<Tensor<1, dim>> cell_velocities(quad.size());
    for (const auto &cell : part.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[FEValuesExtractors::Vector(0)].get_function_values(
            part.get_velocity(), cell_velocities);
          proc_out << cell->active_cell_index() << ": "
                   << fe_values.get_quadrature_points()[0] << ": "
                   << cell_velocities[0] << '\n';
        }
    std::ofstream output;
    if (rank == 0)
      output.open("output");

    print_strings_on_0(proc_out.str(), mpi_comm, output);
  }

  for (auto ptr : u_bc_coefs)
    delete ptr;
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit                      ibtk_init(argc, argv, MPI_COMM_WORLD);
  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "ifed_tag.log");

  test<NDIM>(app_initializer);
}
