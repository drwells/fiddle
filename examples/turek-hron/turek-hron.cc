// Based on the Turek-Hron IBFE benchmark

#include <fiddle/interaction/ifed_method.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/force_contribution_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

std::pair<dealii::Triangulation<2>, dealii::Triangulation<2>>
make_turek_hron_grid(const dealii::Point<2> &cylinder_center = dealii::Point<2>(0.2, 0.2),
                     const double            cylinder_radius = 0.05,
                     const double            beam_height     = 0.02,
                     const double            beam_length     = 0.35,
                     const bool              curve_left_side_of_beam = true)
{
  using namespace dealii;
  AssertThrow(cylinder_radius > beam_height / 2.0,
              ExcMessage("The grid only makes sense when this is true"));
  AssertThrow(beam_length > 0.0,
              ExcMessage("beam length should be positive"));
  // Use the pythagorean theorem to find the location of the beam - here we
  // assume the cylinder is centered at 0, 0
  const Point<2> beam_bottom_left(std::sqrt(std::pow(cylinder_radius, 2) -
                                            std::pow(beam_height / 2.0, 2)),
                                  -beam_height / 2.0);
  const Point<2> beam_upper_right(beam_bottom_left[0] + beam_length,
                                  beam_bottom_left[1] + beam_height);

  // start with the disk grid:
  Triangulation<2> disk;
  GridGenerator::hyper_ball_balanced(disk, Point<2>(), cylinder_radius);
  for (auto &cell : disk.active_cell_iterators())
    cell->set_material_id(0);
  GridTools::rotate(numbers::PI / 8.0, disk);

  for (auto &face : disk.active_face_iterators())
    if (std::abs(face->center(true)[0] - cylinder_radius) < 1e-10)
      {
        if (!curve_left_side_of_beam)
          face->set_manifold_id(numbers::flat_manifold_id);
        // we need to move these vertices down so that they are exactly
        // beam_height apart, yet also still on the disk
        Point<2> &p0 = face->vertex(0);
        Point<2> &p1 = face->vertex(1);
        Assert(std::abs(p0[0] - p1[0]) < 1e-10,
               ExcMessage("x coordinates should be equal on this face"));
        p0[0] = beam_bottom_left[0];
        p0[1] = std::copysign(beam_bottom_left[1], p0[1]);
        p1[0] = beam_bottom_left[0];
        p1[1] = std::copysign(beam_bottom_left[1], p1[1]);
      }

  // now set up the beam:
  Triangulation<2> beam;
  // we need to use one cell in the y direction: try to pick a sane value in x
  const unsigned int n_x_cells = std::max(
    1u, static_cast<unsigned int>(std::round(beam_length / beam_height)));
  GridGenerator::subdivided_hyper_rectangle(beam,
                                            {n_x_cells, 1u},
                                            beam_bottom_left,
                                            beam_upper_right,
                                            true);
  for (auto &cell : beam.active_cell_iterators())
    cell->set_material_id(1);
  GridTools::shift(cylinder_center, disk);
  GridTools::shift(cylinder_center, beam);
  disk.set_manifold(0, PolarManifold<2>(cylinder_center));

  return std::make_pair(std::move(disk), std::move(beam));
}

// Function prototypes
static std::ofstream drag_stream, lift_stream, A_x_posn_stream, A_y_posn_stream;
void
postprocess_data(
  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> patch_hierarchy,
  SAMRAI::tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator,
  const int                                            iteration_num,
  const double                                         loop_time,
  const std::string                                   &data_dump_dirname);

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/

int
main(int argc, char *argv[])
{
  IBTK::IBTKInit ibtk_init(argc, argv);

  {
    using namespace SAMRAI;
    using namespace dealii;
    tbox::Pointer<IBTK::AppInitializer> app_initializer =
      new IBTK::AppInitializer(argc, argv, "IB.log");
    tbox::Pointer<tbox::Database> input_db =
      app_initializer->getInputDatabase();

    // Get various standard options set in the input file.
    const bool dump_viz_data     = app_initializer->dumpVizData();
    const int  viz_dump_interval = app_initializer->getVizDumpInterval();
    const bool uses_visit =
      dump_viz_data && app_initializer->getVisItDataWriter();

    const bool dump_restart_data    = app_initializer->dumpRestartData();
    const int restart_dump_interval = app_initializer->getRestartDumpInterval();
    const std::string restart_dump_dirname =
      app_initializer->getRestartDumpDirectory();

    const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
    const int  postproc_data_dump_interval =
      app_initializer->getPostProcessingDataDumpInterval();
    const std::string postproc_data_dump_dirname =
      app_initializer->getPostProcessingDataDumpDirectory();
    if (dump_postproc_data && (postproc_data_dump_interval > 0) &&
        !postproc_data_dump_dirname.empty())
      {
        tbox::Utilities::recursiveMkdir(postproc_data_dump_dirname);
      }

    const bool dump_timer_data     = app_initializer->dumpTimerData();
    const int  timer_dump_interval = app_initializer->getTimerDumpInterval();

    const double beam_shear_modulus    = input_db->getDouble("beam_shear_modulus");
    const double beam_bulk_modulus     = input_db->getDouble("beam_bulk_modulus");
    const double disk_spring_constant  = input_db->getDouble("disk_spring_constant");
    const double beam_spring_constant  = input_db->getDouble("beam_spring_constant");
    const double disk_damping_constant = input_db->getDouble("disk_damping_constant");

    // Create major algorithm and data objects that comprise the
    // application.  These objects are configured from the input database
    // and, if this is a restarted run, from the restart database.
    tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator =
      new IBAMR::INSStaggeredHierarchyIntegrator(
        "INSStaggeredHierarchyIntegrator",
        app_initializer->getComponentDatabase(
          "INSStaggeredHierarchyIntegrator"));

    // Set up the IFEDMethod.
    const MPI_Comm communicator = IBTK::IBTK_MPI::getCommunicator();
    parallel::shared::Triangulation<2> tria(communicator, {}, true);
    parallel::shared::Triangulation<1, 2> boundary_tria(communicator, {}, true);
    const Point<2> cylinder_center(0.2, 0.2);
    auto pair = ModelData::make_turek_hron_grid(cylinder_center, 0.05, 0.02, 0.35, false);
    auto &disk = pair.first;
    auto &beam = pair.second;
    const bool use_penalty_cylinder = input_db->getBoolWithDefault("use_penalty_cylinder", false);

    const unsigned int penalty_fe_degree = 1;
    const unsigned int beam_fe_degree = 2;
    FESystem<2>        fe(FE_Q<2>(beam_fe_degree), 2);
    FESystem<1, 2>     penalty_fe(FE_Q<1, 2>(penalty_fe_degree), 2);

    QGauss<1> boundary_quadrature(fe.tensor_degree() + 1);
    QGauss<2> quadrature1(fe.tensor_degree() + 1);
    QGauss<2> quadrature2(fe.tensor_degree() + 2);

    std::vector<types::material_id> penalty_ids;
    std::vector<types::material_id> beam_ids;
    // We can skip the ID lookup when we have separate parts
    if (use_penalty_cylinder)
      {
        penalty_ids.push_back(0);
        beam_ids.push_back(1);
      }

    std::vector<std::unique_ptr<fdl::ForceContribution<2>>> beam_forces;
    beam_forces.emplace_back(std::make_unique<fdl::ModifiedNeoHookeanStress<2>>(
      quadrature2, beam_shear_modulus, beam_ids));
    beam_forces.emplace_back(std::make_unique<fdl::JLogJVolumetricEnergyStress<2>>(
      quadrature2, beam_bulk_modulus, beam_ids));
    std::vector<std::unique_ptr<fdl::ForceContribution<1, 2>>> penalty_forces;

    std::vector<fdl::Part<2>> parts;
    std::vector<fdl::Part<1, 2>> penalty_parts;
    const double target_element_size =
      input_db->getDouble("MFAC") * input_db->getDouble("DX");
    if (use_penalty_cylinder)
      {
        while (GridTools::maximal_cell_diameter(disk) > target_element_size * penalty_fe_degree)
          disk.refine_global(1);
        while (GridTools::maximal_cell_diameter(beam) > target_element_size * beam_fe_degree)
          beam.refine_global(1);

        std::vector<types::boundary_id> left{0u};
        beam_forces.emplace_back(std::make_unique<fdl::BoundarySpringForce<2>>(
          boundary_quadrature, beam_spring_constant, left));

        boundary_tria.set_manifold(0, PolarManifold<1, 2>(cylinder_center));
        GridGenerator::extract_boundary_mesh(
          disk, static_cast<Triangulation<1, 2> &>(boundary_tria));
        penalty_forces.emplace_back(std::make_unique<fdl::SpringForce<1, 2>>(
          boundary_quadrature, disk_spring_constant, penalty_ids));
        penalty_forces.emplace_back(std::make_unique<fdl::DampingForce<1, 2>>(
          boundary_quadrature, disk_damping_constant));
        penalty_parts.emplace_back(boundary_tria, penalty_fe, std::move(penalty_forces));
        tria.copy_triangulation(beam);
      }
    else
      {
        GridGenerator::merge_triangulations(pair.first, pair.second, tria, 1e-10, true, true);
        tria.set_manifold(0, PolarManifold<2>(cylinder_center));

        while (GridTools::maximal_cell_diameter(tria) > beam_fe_degree * target_element_size)
          tria.refine_global(1);
        // TODO: material-id dependent damping is not yet implemented
        // std::vector<types::material_id> disk_mids({0u});
        // beam_forces.emplace_back(std::make_unique<fdl::DampingForce<2>>(
        //   quadrature1, disk_damping_constant, disk_mids));
        beam_forces.emplace_back(std::make_unique<fdl::SpringForce<2>>(
          quadrature1, disk_spring_constant, penalty_ids));
      }
    parts.emplace_back(tria, fe, std::move(beam_forces));
    
    tbox::Pointer<fdl::IFEDMethod<2>> ib_method_ops =
      new fdl::IFEDMethod<2>("IFEDMethod",
                             app_initializer->getComponentDatabase(
                               "IFEDMethod"),
                             std::move(penalty_parts),
                             std::move(parts));
    tbox::Pointer<IBAMR::IBHierarchyIntegrator> time_integrator =
      new IBAMR::IBExplicitHierarchyIntegrator(
        "IBHierarchyIntegrator",
        app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
        ib_method_ops,
        navier_stokes_integrator);
    tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
      new geom::CartesianGridGeometry<NDIM>(
        "CartesianGeometry",
        app_initializer->getComponentDatabase("CartesianGeometry"));
    tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy =
      new hier::PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
    tbox::Pointer<mesh::StandardTagAndInitialize<NDIM>> error_detector =
      new mesh::StandardTagAndInitialize<NDIM>(
        "StandardTagAndInitialize",
        time_integrator,
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

    // Create Eulerian initial condition specification objects.
    if (input_db->keyExists("VelocityInitialConditions"))
      {
        tbox::Pointer<IBTK::CartGridFunction> u_init =
          new IBTK::muParserCartGridFunction(
            "u_init",
            app_initializer->getComponentDatabase("VelocityInitialConditions"),
            grid_geometry);
        navier_stokes_integrator->registerVelocityInitialConditions(u_init);
      }

    if (input_db->keyExists("PressureInitialConditions"))
      {
        tbox::Pointer<IBTK::CartGridFunction> p_init =
          new IBTK::muParserCartGridFunction(
            "p_init",
            app_initializer->getComponentDatabase("PressureInitialConditions"),
            grid_geometry);
        navier_stokes_integrator->registerPressureInitialConditions(p_init);
      }

    // Create Eulerian boundary condition specification objects (when
    // necessary).
    const hier::IntVector<NDIM> &periodic_shift =
      grid_geometry->getPeriodicShift();
    std::vector<solv::RobinBcCoefStrategy<NDIM> *> u_bc_coefs(NDIM);
    if (periodic_shift.min() > 0)
      {
        for (unsigned int d = 0; d < NDIM; ++d)
          {
            u_bc_coefs[d] = NULL;
          }
      }
    else
      {
        for (unsigned int d = 0; d < NDIM; ++d)
          {
            std::ostringstream bc_coefs_name_stream;
            bc_coefs_name_stream << "u_bc_coefs_" << d;
            const std::string bc_coefs_name = bc_coefs_name_stream.str();

            std::ostringstream bc_coefs_db_name_stream;
            bc_coefs_db_name_stream << "VelocityBcCoefs_" << d;
            const std::string bc_coefs_db_name = bc_coefs_db_name_stream.str();

            u_bc_coefs[d] = new IBTK::muParserRobinBcCoefs(
              bc_coefs_name,
              app_initializer->getComponentDatabase(bc_coefs_db_name),
              grid_geometry);
          }
        navier_stokes_integrator->registerPhysicalBoundaryConditions(
          u_bc_coefs);
      }

    // Create Eulerian body force function specification objects.
    if (input_db->keyExists("ForcingFunction"))
      {
        tbox::Pointer<IBTK::CartGridFunction> f_fcn =
          new IBTK::muParserCartGridFunction(
            "f_fcn",
            app_initializer->getComponentDatabase("ForcingFunction"),
            grid_geometry);
        time_integrator->registerBodyForceFunction(f_fcn);
      }

    // Set up visualization plot file writers.
    tbox::Pointer<appu::VisItDataWriter<NDIM>> visit_data_writer =
      app_initializer->getVisItDataWriter();
    if (uses_visit)
      {
        time_integrator->registerVisItDataWriter(visit_data_writer);
      }

    // Initialize hierarchy configuration and data on all patches.
    time_integrator->initializePatchHierarchy(patch_hierarchy,
                                              gridding_algorithm);

    // Print the input database contents to the log file.
    tbox::plog << "Input database:\n";
    input_db->printClassData(tbox::plog);

    // Write out initial visualization data.
    int    iteration_num = time_integrator->getIntegratorStep();
    double loop_time     = time_integrator->getIntegratorTime();
    if (dump_viz_data)
      {
        tbox::pout << "\n\nWriting visualization files...\n\n";
        if (uses_visit)
          {
            time_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy,
                                             iteration_num,
                                             loop_time);
          }

        if (ib_method_ops->n_penalty_parts() > 0)
          {
            const auto &part = ib_method_ops->get_penalty_part(0);
            DataOut<1, 2>  data_out;
            data_out.attach_dof_handler(part.get_dof_handler());
            data_out.add_data_vector(part.get_velocity(), "U");

            MappingFEField<1, 2, LinearAlgebra::distributed::Vector<double>>
              position_mapping(part.get_dof_handler(), part.get_position());
            data_out.build_patches(position_mapping);
            data_out.write_vtu_with_pvtu_record(
              app_initializer->getVizDumpDirectory() + "/",
              "penalty",
              iteration_num,
              IBTK::IBTK_MPI::getCommunicator(),
              8);
          }

        const auto &part = ib_method_ops->get_part(0);
        DataOut<2>  data_out;
        data_out.attach_dof_handler(part.get_dof_handler());
        data_out.add_data_vector(part.get_velocity(), "U");

        MappingFEField<2, 2, LinearAlgebra::distributed::Vector<double>>
          position_mapping(part.get_dof_handler(), part.get_position());
        data_out.build_patches(position_mapping);
        data_out.write_vtu_with_pvtu_record(
          app_initializer->getVizDumpDirectory() + "/",
          "part",
          iteration_num,
          IBTK::IBTK_MPI::getCommunicator(),
          8);
      }

    // Open streams to save lift and drag coefficients.
    if (tbox::SAMRAI_MPI::getRank() == 0)
      {
        drag_stream.open("C_D.curve",
                         std::ios_base::out | std::ios_base::trunc);
        lift_stream.open("C_L.curve",
                         std::ios_base::out | std::ios_base::trunc);
        A_x_posn_stream.open("A_x.curve",
                             std::ios_base::out | std::ios_base::trunc);
        A_y_posn_stream.open("A_y.curve",
                             std::ios_base::out | std::ios_base::trunc);
      }

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

        // At specified intervals, write visualization and restart files,
        // print out timer data, and store hierarchy data for post
        // processing.
        iteration_num += 1;
        const bool last_step = !time_integrator->stepsRemaining();
        if (dump_viz_data &&
            (iteration_num % viz_dump_interval == 0 || last_step))
          {
            tbox::pout << "\nWriting visualization files...\n\n";
            if (uses_visit)
              {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy,
                                                 iteration_num,
                                                 loop_time);
              }


            if (ib_method_ops->n_penalty_parts() > 0)
              {
                const auto &part = ib_method_ops->get_penalty_part(0);
                DataOut<1, 2>  data_out;
                data_out.attach_dof_handler(part.get_dof_handler());
                data_out.add_data_vector(part.get_velocity(), "U");
     
                MappingFEField<1, 2, LinearAlgebra::distributed::Vector<double>>
                  position_mapping(part.get_dof_handler(), part.get_position());
                data_out.build_patches(position_mapping);
                data_out.write_vtu_with_pvtu_record(
                  app_initializer->getVizDumpDirectory() + "/",
                  "penalty",
                  iteration_num,
                  IBTK::IBTK_MPI::getCommunicator(),
                  8);
              }

            const auto &part = ib_method_ops->get_part(0);
            DataOut<2>  data_out;
            data_out.attach_dof_handler(part.get_dof_handler());
            data_out.add_data_vector(part.get_velocity(), "U");

            MappingFEField<2, 2, LinearAlgebra::distributed::Vector<double>>
              position_mapping(part.get_dof_handler(), part.get_position());
            data_out.build_patches(position_mapping);
            data_out.write_vtu_with_pvtu_record(
              app_initializer->getVizDumpDirectory() + "/",
              "part",
              iteration_num,
              IBTK::IBTK_MPI::getCommunicator(),
              8);
          }
        if (dump_restart_data &&
            (iteration_num % restart_dump_interval == 0 || last_step))
          {
            tbox::pout << "\nWriting restart files...\n\n";
            tbox::RestartManager::getManager()->writeRestartFile(
              restart_dump_dirname, iteration_num);
          }
        if (dump_timer_data &&
            (iteration_num % timer_dump_interval == 0 || last_step))
          {
            tbox::pout << "\nWriting timer data...\n\n";
            tbox::TimerManager::getManager()->print(tbox::plog);
          }
        if (dump_postproc_data &&
            (iteration_num % postproc_data_dump_interval == 0 || last_step))
          {
            tbox::pout << "\nWriting state data...\n\n";
            // postprocess_data();
          }
      }

    // Close the logging streams.
    if (tbox::SAMRAI_MPI::getRank() == 0)
      {
        drag_stream.close();
        lift_stream.close();
        A_x_posn_stream.close();
        A_y_posn_stream.close();
      }

    // Cleanup Eulerian boundary condition specification objects (when
    // necessary).
    for (unsigned int d = 0; d < NDIM; ++d)
      delete u_bc_coefs[d];

  } // cleanup dynamically allocated objects prior to shutdown

  SAMRAI::tbox::SAMRAIManager::shutdown();
}

void
postprocess_data(
  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> /*patch_hierarchy*/,
  SAMRAI::tbox::Pointer<
    IBAMR::INSHierarchyIntegrator> /*navier_stokes_integrator*/,
  const int /*iteration_num*/,
  const double /*loop_time*/,
  const std::string & /*data_dump_dirname*/)
{
  // compute the lift and drag, somehow
} // postprocess_data
