#include <fiddle/interaction/ifed_method.h>

#include <fiddle/mechanics/force_contribution_lib.h>
#include <fiddle/mechanics/part.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <memory>
#include <string>
#include <vector>

using namespace dealii;
using namespace SAMRAI;

/**
 * Deviatoric stress for the band.
 *
 * This is exactly the version used in the input file for the IBFE elastic
 * band benchmark (i.e., PK1_dev_stress_function_mod, which does not used
 * modified invariants). We may want to switch to using modified invariants.
 */
class BandDeviatoricStress : public fdl::ForceContribution<2>
{
public:
  BandDeviatoricStress(const Quadrature<2> &quad, const double shear_modulus)
    : ForceContribution<2>(quad)
    , shear_modulus(shear_modulus)
  {}

  virtual bool
  is_stress() const override
  {
    return true;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::update_FF | fdl::update_FF_inv_T;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual void
  compute_stress(const double                   /*time*/,
                 const fdl::MechanicsValues<2> &me_values,
                 const typename Triangulation<2>::active_cell_iterator &/*cell*/,
                 ArrayView<Tensor<2, 2>> &stresses) const override
  {
    for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
      {
        const auto &FF       = me_values.get_FF()[qp_n];
        const auto &FF_inv_T = me_values.get_FF_inv_T()[qp_n];
        stresses[qp_n]       = shear_modulus * (FF - FF_inv_T);
      }
  }

private:
  double shear_modulus;
};

/**
 * Dilatational stress for the band.
 *
 * This is exactly the version used in the input file for the IBFE elastic
 * band benchmark (PK1_dil_stress_function).
 */
class BandDilatationalStress : public fdl::ForceContribution<2>
{
public:
  BandDilatationalStress(const Quadrature<2> &quad, const double bulk_modulus)
    : ForceContribution<2>(quad)
    , bulk_modulus(bulk_modulus)
  {}

  virtual bool
  is_stress() const override
  {
    return true;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::update_FF | fdl::update_det_FF | fdl::update_FF_inv_T;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual void
  compute_stress(const double                   /*time*/,
                 const fdl::MechanicsValues<2> &me_values,
                 const typename Triangulation<2>::active_cell_iterator &/*cell*/,
                 ArrayView<Tensor<2, 2>> &stresses) const override
  {
    for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
      {
        const auto &FF_inv_T = me_values.get_FF_inv_T()[qp_n];
        const auto &det_FF   = me_values.get_det_FF()[qp_n];
        stresses[qp_n]       = bulk_modulus * std::log(det_FF) * FF_inv_T;
      }
  }

private:
  double bulk_modulus;
};

/**
 * Stress function for the block. Same as PK1_block_stress_function() in the
 * original program.
 */
class BlockStress : public fdl::ForceContribution<2>
{
public:
  BlockStress(const Quadrature<2> &quad, const double kappa)
    : ForceContribution<2>(quad)
    , kappa(kappa)
  {}

  virtual bool
  is_stress() const override
  {
    return true;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::update_FF | fdl::update_FF_inv_T;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual void
  compute_stress(const double                   /*time*/,
                 const fdl::MechanicsValues<2> &me_values,
                 const typename Triangulation<2>::active_cell_iterator &/*cell*/,
                 ArrayView<Tensor<2, 2>> &stresses) const override
  {
    for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
      {
        const auto &FF       = me_values.get_FF()[qp_n];
        const auto &FF_inv_T = me_values.get_FF_inv_T()[qp_n];
        stresses[qp_n]       = kappa * (FF - FF_inv_T);
      }
  }

private:
  double kappa;
};

/*
 * Postprocessor showing the Jacobian of the present position field.
 */
class JacobianPostprocessor : public DataPostprocessorScalar<2>
{
public:
    JacobianPostprocessor()
        : DataPostprocessorScalar<2>("Jacobian", UpdateFlags::update_gradients)
    {}

    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<2> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override
    {

    Assert(computed_quantities.size() == inputs.solution_values.size(),
           ExcDimensionMismatch(computed_quantities.size(),
                                inputs.solution_values.size()));
    for (unsigned int i = 0; i < computed_quantities.size(); i++)
      {
        Assert(computed_quantities[i].size() == 1,
               ExcDimensionMismatch(computed_quantities[i].size(), 1));

        Tensor<2, 2> FF;
        FF[0] = inputs.solution_gradients[i][0];
        FF[1] = inputs.solution_gradients[i][1];
        computed_quantities[i][0] = determinant(FF);
      }
    }
};

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
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);

  { // cleanup dynamically allocated objects prior to shutdown

    // Parse command line options, set some standard options from the input
    // file, initialize the restart database (if this is a restarted run),
    // and enable file logging.
    tbox::Pointer<IBTK::AppInitializer> app_initializer =
      new IBTK::AppInitializer(argc, argv, "IB.log");
    tbox::Pointer<tbox::Database> input_db =
      app_initializer->getInputDatabase();

    // Get various standard options set in the input file.
    const bool dump_viz_data     = app_initializer->dumpVizData();
    const int  viz_dump_interval = app_initializer->getVizDumpInterval();
    const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();

    const bool dump_restart_data    = app_initializer->dumpRestartData();
    const int restart_dump_interval = app_initializer->getRestartDumpInterval();
    const std::string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

    const bool dump_timer_data     = app_initializer->dumpTimerData();
    const int  timer_dump_interval = app_initializer->getTimerDumpInterval();

    //
    // Create relevant triangulations
    //
    auto         fe_db = input_db->getDatabase("FiniteElementModel");
    const double target_element_size =
      fe_db->getDouble("MFAC") * input_db->getDouble("DX") * std::sqrt(2);
    auto refine_tria = [&](Triangulation<2> &tria) {
      while (GridTools::maximal_cell_diameter(tria) > target_element_size)
        tria.refine_global(1);
    };
    const MPI_Comm communicator = tbox::SAMRAI_MPI::getCommunicator();

    parallel::shared::Triangulation<2> band(communicator, {}, true);
    GridGenerator::subdivided_hyper_rectangle(band,
                                              {1u, 9u},
                                              Point<2>(0.95, 0.05),
                                              Point<2>(1.05, 0.95),
                                              /*colorize=*/true);

    parallel::shared::Triangulation<2> lower_block(communicator, {}, true);
    GridGenerator::hyper_rectangle(lower_block,
                                   Point<2>(0.95, 0.0),
                                   Point<2>(1.05, 0.1),
                                   /*colorize=*/true);

    parallel::shared::Triangulation<2> upper_block(communicator, {}, true);
    GridGenerator::hyper_rectangle(upper_block,
                                   Point<2>(0.95, 0.9),
                                   Point<2>(1.05, 1.0),
                                   /*colorize=*/true);

    refine_tria(band);
    refine_tria(lower_block);
    refine_tria(upper_block);

    //
    // Set up the remaining IFED mechanics components:
    //
    FESystem<2>               fe(FE_Q<2>(fe_db->getInteger("fe_degree")), 2);
    std::vector<fdl::Part<2>> parts;
    {
      std::shared_ptr<DoFHandler<2>> band_dof_handler =
        std::make_shared<DoFHandler<2>>(band);
      band_dof_handler->distribute_dofs(fe);
      std::shared_ptr<DoFHandler<2>> lower_block_dof_handler =
        std::make_shared<DoFHandler<2>>(lower_block);
      lower_block_dof_handler->distribute_dofs(fe);
      std::shared_ptr<DoFHandler<2>> upper_block_dof_handler =
        std::make_shared<DoFHandler<2>>(upper_block);
      upper_block_dof_handler->distribute_dofs(fe);

      QGauss<2> body_force_quad(fe.tensor_degree() + 1);
      QGauss<1> boundary_force_quad(fe.tensor_degree() + 1);

      QGauss<2> dev_quad(
        fe_db->getIntegerWithDefault("pk1_dev_n_points_1d", 2));
      QGauss<2> dil_quad(
        fe_db->getIntegerWithDefault("pk1_dil_n_points_1d", 1));

      auto material_db = input_db->getDatabase("MaterialModel");
      std::vector<std::unique_ptr<fdl::ForceContribution<2>>> band_forces;
      // band has top and bottom tether forces, a damping force, and an elastic
      // model.
      band_forces.emplace_back(new fdl::BoundarySpringForce<2>(
        boundary_force_quad,
        material_db->getDouble("band_spring_force_coefficient"),
        *band_dof_handler,
        MappingQ<2>(1),
        {types::boundary_id(2), types::boundary_id(3)},
        Functions::IdentityFunction<2>()));
      band_forces.emplace_back(new fdl::DampingForce<2>(
        body_force_quad, material_db->getDouble("band_damping_coefficient")));
      band_forces.emplace_back(
        new BandDeviatoricStress(dev_quad,
                                 material_db->getDouble("band_shear_modulus")));
      band_forces.emplace_back(new BandDilatationalStress(
        dil_quad, material_db->getDouble("band_bulk_modulus")));

      // lower block has tether and a stress
      std::vector<std::unique_ptr<fdl::ForceContribution<2>>>
        lower_block_forces;
      lower_block_forces.emplace_back(new fdl::SpringForce<2>(
        body_force_quad,
        material_db->getDouble("block_spring_force_coefficient"),
        *lower_block_dof_handler,
        MappingQ<2>(1),
        Functions::IdentityFunction<2>()));
      lower_block_forces.emplace_back(
        new BlockStress(dev_quad, material_db->getDouble("block_kappa")));

      // upper block has tether and a stress
      std::vector<std::unique_ptr<fdl::ForceContribution<2>>>
        upper_block_forces;
      upper_block_forces.emplace_back(new fdl::SpringForce<2>(
        body_force_quad,
        material_db->getDouble("block_spring_force_coefficient"),
        *upper_block_dof_handler,
        MappingQ<2>(1),
        Functions::IdentityFunction<2>()));
      upper_block_forces.emplace_back(
        new BlockStress(dev_quad, material_db->getDouble("block_kappa")));

      parts.emplace_back(band_dof_handler, std::move(band_forces));
      parts.emplace_back(lower_block_dof_handler,
                         std::move(lower_block_forces));
      parts.emplace_back(upper_block_dof_handler,
                         std::move(upper_block_forces));
    }

    tbox::Pointer<fdl::IFEDMethod<2>> ib_method_ops =
      new fdl::IFEDMethod<2>("IFEDMethod",
                             app_initializer->getComponentDatabase(
                               "IFEDMethod"),
                             std::move(parts));

    //
    // Set up IBAMR and SAMRAI data structures:
    //
    tbox::Pointer<IBAMR::INSHierarchyIntegrator> navier_stokes_integrator;
    const std::string                            solver_type =
      app_initializer->getComponentDatabase("Main")->getString("solver_type");
    if (solver_type == "STAGGERED")
      {
        navier_stokes_integrator = new IBAMR::INSStaggeredHierarchyIntegrator(
          "INSStaggeredHierarchyIntegrator",
          app_initializer->getComponentDatabase(
            "INSStaggeredHierarchyIntegrator"));
      }
    else if (solver_type == "COLLOCATED")
      {
        navier_stokes_integrator = new IBAMR::INSCollocatedHierarchyIntegrator(
          "INSCollocatedHierarchyIntegrator",
          app_initializer->getComponentDatabase(
            "INSCollocatedHierarchyIntegrator"));
      }
    else
      {
        TBOX_ERROR("Unsupported solver type: "
                   << solver_type << "\n"
                   << "Valid options are: COLLOCATED, STAGGERED");
      }

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
            u_bc_coefs[d] = new IBTK::muParserRobinBcCoefs(
              "u_bc_coefs" + std::to_string(d),
              app_initializer->getComponentDatabase("VelocityBcCoefs_" +
                                                    std::to_string(d)),
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

    auto write_fe_output = [&]() {
      for (unsigned int part_n = 0; part_n < ib_method_ops->n_parts(); ++part_n)
        {
          const auto & part = ib_method_ops->get_part(part_n);
          JacobianPostprocessor postprocessor;

          DataOut<2> data_out;
          data_out.attach_dof_handler(part.get_dof_handler());
          data_out.add_data_vector(part.get_velocity(), "U");

          MappingFEField<2, 2, LinearAlgebra::distributed::Vector<double>>
            position_mapping(part.get_dof_handler(), part.get_position());
          data_out.add_data_vector(part.get_position(), postprocessor);
          data_out.build_patches(position_mapping);

          data_out.write_vtu_with_pvtu_record(app_initializer->getVizDumpDirectory() +
                                                "/",
                                              "structure-" + std::to_string(part_n),
                                              iteration_num,
                                              communicator,
                                              8);
        }
    };
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

        write_fe_output();
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

            write_fe_output();
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
      }

    // Determine the accuracy of the computed solution.
    tbox::pout << "\n"
               << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n"
               << "Computing error norms.\n\n";
    const int              finest_ln = patch_hierarchy->getFinestLevelNumber();
    IBTK::HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy);
    hier_math_ops.resetLevels(finest_ln, finest_ln);
    hier::VariableDatabase<NDIM> *var_db =
      hier::VariableDatabase<NDIM>::getDatabase();
    tbox::Pointer<hier::Variable<NDIM>> u_var =
      time_integrator->getVelocityVariable();
    const tbox::Pointer<hier::VariableContext> u_ctx =
      time_integrator->getCurrentContext();
    const int u_idx       = var_db->mapVariableAndContextToIndex(u_var, u_ctx);
    const int coarsest_ln = 0;
    hier_math_ops.setPatchHierarchy(patch_hierarchy);
    hier_math_ops.resetLevels(coarsest_ln, finest_ln);
    const int wgt_sc_idx = hier_math_ops.getSideWeightPatchDescriptorIndex();
    math::HierarchySideDataOpsReal<NDIM, double> hier_sc_data_ops(
      patch_hierarchy, coarsest_ln, finest_ln);
    tbox::pout << std::setprecision(16) << "Error in u at time " << loop_time
               << ":\n"
               << "  L1-norm:  " << hier_sc_data_ops.L1Norm(u_idx, wgt_sc_idx)
               << "\n"
               << "  L2-norm:  " << hier_sc_data_ops.L2Norm(u_idx, wgt_sc_idx)
               << "\n"
               << "  max-norm: " << hier_sc_data_ops.maxNorm(u_idx, wgt_sc_idx)
               << "\n"
               << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    // Cleanup Eulerian boundary condition specification objects (when
    // necessary).
    for (unsigned int d = 0; d < NDIM; ++d)
      delete u_bc_coefs[d];

  } // cleanup dynamically allocated objects prior to shutdown

  tbox::SAMRAIManager::shutdown();
  return 0;
}
