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

// Elasticity model data.
namespace ModelData
{
  using namespace dealii;

  void
  make_turek_hron_grid(Triangulation<2> &tria,
                       const Point<2> &  cylinder_center = Point<2>(0.2, 0.2),
                       const double      cylinder_radius = 0.05,
                       const double      tail_height     = 0.02,
                       const double      tail_length     = 0.35,
                       const bool        curve_left_side_of_tail = true)
  {
    AssertThrow(cylinder_radius > tail_height / 2.0,
                ExcMessage("The grid only makes sense when this is true"));
    AssertThrow(tail_length > 0.0,
                ExcMessage("tail length should be positive"));
    // Use the pythagorean theorem to find the location of the tail - here we
    // assume the cylinder is centered at 0, 0
    const Point<2> tail_bottom_left(std::sqrt(std::pow(cylinder_radius, 2) -
                                              std::pow(tail_height / 2.0, 2)),
                                    -tail_height / 2.0);
    const Point<2> tail_upper_right(tail_bottom_left[0] + tail_length,
                                    tail_bottom_left[1] + tail_height);

    // start with the disk grid:
    Triangulation<2> disk;
    GridGenerator::hyper_ball_balanced(disk, Point<2>(), cylinder_radius);
    for (auto &cell : disk.active_cell_iterators())
      cell->set_material_id(0);
    GridTools::rotate(numbers::PI / 8.0, disk);

    for (auto &face : disk.active_face_iterators())
      if (std::abs(face->center(true)[0] - cylinder_radius) < 1e-10)
        {
          if (!curve_left_side_of_tail)
            face->set_manifold_id(numbers::flat_manifold_id);
          // we need to move these vertices down so that they are exactly
          // tail_height apart, yet also still on the disk
          Point<2> &p0 = face->vertex(0);
          Point<2> &p1 = face->vertex(1);
          Assert(std::abs(p0[0] - p1[0]) < 1e-10,
                 ExcMessage("x coordinates should be equal on this face"));
          p0[0] = tail_bottom_left[0];
          p0[1] = std::copysign(tail_bottom_left[1], p0[1]);
          p1[0] = tail_bottom_left[0];
          p1[1] = std::copysign(tail_bottom_left[1], p1[1]);
        }

    // now set up the tail:
    Triangulation<2> tail;
    // we need to use one cell in the y direction: try to pick a sane value in x
    const unsigned int n_x_cells = std::max(
      1u, static_cast<unsigned int>(std::round(tail_length / tail_height)));
    GridGenerator::subdivided_hyper_rectangle(tail,
                                              {n_x_cells, 1u},
                                              tail_bottom_left,
                                              tail_upper_right);
    for (auto &cell : tail.active_cell_iterators())
      cell->set_material_id(1);
    GridGenerator::merge_triangulations(disk, tail, tria, 1.0e-12, true);
    GridTools::shift(cylinder_center, tria);
    tria.set_manifold(0, PolarManifold<2>(cylinder_center));
  }

  class BeamNeoHookeanStress : public fdl::ForceContribution<2>
  {
  public:
    // todo more parameters
    BeamNeoHookeanStress(const Quadrature<2> &    quad,
                         const types::material_id beam_id,
                         const double             mu_s)
      : ForceContribution<2>(quad)
      , beam_id(beam_id)
      , mu_s(mu_s)
    {}

    virtual bool
    is_stress() const override
    {
      return true;
    }

    virtual fdl::MechanicsUpdateFlags
    get_mechanics_update_flags() const override
    {
      return fdl::update_det_FF | fdl::update_FF | fdl::update_FF_inv_T |
             fdl::update_first_invariant;
    }

    virtual UpdateFlags
    get_update_flags() const override
    {
      return UpdateFlags::update_default;
    }

    virtual void
    compute_stress(const double                   time,
                   const fdl::MechanicsValues<2> &me_values,
                   const typename Triangulation<2>::active_cell_iterator &cell,
                   ArrayView<Tensor<2, 2>> &stresses) const override
    {
      if (cell->material_id() == beam_id)
        {
          for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
            {
              const auto &J        = me_values.get_det_FF()[qp_n];
              const auto &FF       = me_values.get_FF()[qp_n];
              const auto &I1       = me_values.get_first_invariant()[qp_n];
              const auto &FF_inv_T = me_values.get_FF_inv_T()[qp_n];
              // TODO make sure this is right in true 2D
              stresses[qp_n] = mu_s / J * (FF - (I1 / 2.0) * FF_inv_T);
            }
        }
      else
        {
          std::fill(stresses.begin(), stresses.end(), Tensor<2, 2>());
        }
    }

  private:
    types::material_id beam_id;
    double             mu_s;
  };

  class BeamDilatationalStress : public fdl::ForceContribution<2>
  {
  public:
    BeamDilatationalStress(const Quadrature<2> &    quad,
                           const types::material_id beam_id,
                           const double             beta_s)
      : ForceContribution<2>(quad)
      , beam_id(beam_id)
      , beta_s(beta_s)
    {}

    virtual bool
    is_stress() const override
    {
      return true;
    }

    virtual fdl::MechanicsUpdateFlags
    get_mechanics_update_flags() const override
    {
      return fdl::update_det_FF | fdl::update_FF_inv_T;
    }

    virtual UpdateFlags
    get_update_flags() const override
    {
      return UpdateFlags::update_default;
    }

    virtual void
    compute_stress(const double                   time,
                   const fdl::MechanicsValues<2> &me_values,
                   const typename Triangulation<2>::active_cell_iterator &cell,
                   ArrayView<Tensor<2, 2>> &stresses) const override
    {
      // if (cell->material_id() == beam_id)
      if (true)
        {
          for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
            {
              const auto &J        = me_values.get_det_FF()[qp_n];
              const auto &FF_inv_T = me_values.get_FF_inv_T()[qp_n];
              stresses[qp_n]       = beta_s * J * std::log(J) * FF_inv_T;
            }
        }
      else
        {
          std::fill(stresses.begin(), stresses.end(), Tensor<2, 2>());
        }
    }

  private:
    types::material_id beam_id;
    double             beta_s;
  };
} // namespace ModelData

// Function prototypes
static std::ofstream drag_stream, lift_stream, A_x_posn_stream, A_y_posn_stream;
void
postprocess_data(
  SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> patch_hierarchy,
  SAMRAI::tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator,
  const int                                            iteration_num,
  const double                                         loop_time,
  const std::string &                                  data_dump_dirname);

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

    const double dx = input_db->getDouble("DX");
    const double ds = input_db->getDouble("MFAC") * dx;

    const double mu_s          = input_db->getDouble("MU_S");
    const double beta_s        = input_db->getDouble("BETA_S");
    const double kappa_s_block = input_db->getDouble("KAPPA_S_BLOCK");

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
    dealii::parallel::shared::Triangulation<2> tria(communicator, {}, true);
    ModelData::make_turek_hron_grid(tria);

    // include EFAC
    const double target_element_size =
      input_db->getDouble("MFAC") * input_db->getDouble("DX");
    while (dealii::GridTools::maximal_cell_diameter(tria) > target_element_size)
      tria.refine_global(1);

    dealii::FESystem<2> fe(dealii::FE_Q<2>(2), 2);
    dealii::QGauss<2>   quadrature1(fe.tensor_degree() + 1);
    dealii::QGauss<2>   quadrature2(fe.tensor_degree() + 2);
    auto dof_handler = std::make_shared<dealii::DoFHandler<2>>(tria);
    dof_handler->distribute_dofs(fe);

    const std::vector<dealii::types::material_id> cylinder_ids{0};

    MappingQ<dim, spacedim> mapping(fe.tensor_degree());
    auto spring_force = std::make_unique<fdl::SpringForce<2>>(
      quadrature1,
      kappa_s_block,
      *dof_handler,
      mapping,
      cylinder_ids,
      dealii::Functions::IdentityFunction<2>());

    std::vector<std::unique_ptr<fdl::ForceContribution<2>>> force_contributions;
    force_contributions.emplace_back(std::move(spring_force));
    force_contributions.emplace_back(
      std::make_unique<ModelData::BeamNeoHookeanStress>(quadrature2, 1, mu_s));
    force_contributions.emplace_back(
      std::make_unique<ModelData::BeamDilatationalStress>(quadrature2,
                                                          1,
                                                          beta_s));

    std::vector<fdl::Part<2>> parts;
    parts.emplace_back(tria, fe, std::move(force_contributions));
    tbox::Pointer<fdl::IFEDMethod<2>> ib_method_ops =
      new fdl::IFEDMethod<2>("IFEDMethod",
                             app_initializer->getComponentDatabase(
                               "IFEDMethod"),
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

    const bool from_restart =
      tbox::RestartManager::getManager()->isFromRestart();

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

        using namespace dealii;
        const auto &part = ib_method_ops->get_part(0);
        DataOut<2>  data_out;
        data_out.attach_dof_handler(part.get_dof_handler());
        data_out.add_data_vector(part.get_velocity(), "U");

        MappingFEField<2, 2, LinearAlgebra::distributed::Vector<double>>
          position_mapping(part.get_dof_handler(), part.get_position());
        data_out.build_patches(position_mapping);
        data_out.write_vtu_with_pvtu_record(
          app_initializer->getVizDumpDirectory() + "/",
          "solution",
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

            using namespace dealii;
            const auto &part = ib_method_ops->get_part(0);
            DataOut<2>  data_out;
            data_out.attach_dof_handler(part.get_dof_handler());
            data_out.add_data_vector(part.get_velocity(), "U");

            MappingFEField<2, 2, LinearAlgebra::distributed::Vector<double>>
              position_mapping(part.get_dof_handler(), part.get_position());
            data_out.build_patches(position_mapping);
            data_out.write_vtu_with_pvtu_record(
              app_initializer->getVizDumpDirectory() + "/",
              "solution",
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
  const double loop_time,
  const std::string & /*data_dump_dirname*/)
{
  // compute the lift and drag, somehow
} // postprocess_data
