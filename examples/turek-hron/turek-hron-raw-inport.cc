// Based on the Turek-Hron IBFE benchmark

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for basic libMesh objects
#include <libmesh/boundary_info.h>
#include <libmesh/boundary_mesh.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_function.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_triangle_interface.h>

// Headers for application-specific algorithm/data structure objects
#include <boost/multi_array.hpp>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>

// Elasticity model data.
namespace ModelData
{
  // Tether (penalty) force function for the solid block.
  static double kappa_s_block = 1.0e6;
  static double eta_s_block   = 0.0;
  void
  block_tether_force_function(
    libMesh::VectorValue<double> &F,
    const libMesh::TensorValue<double> & /*FF*/,
    const libMesh::Point &x,
    const libMesh::Point &X,
    libMesh::Elem *const /*elem*/,
    const std::vector<const std::vector<double> *> &var_data,
    const std::vector<const std::vector<libMesh::VectorValue<double>> *>
      & /*grad_var_data*/,
    double /*time*/,
    void * /*ctx*/)
  {
    const std::vector<double> &U = *var_data[0];
    for (unsigned int d = 0; d < NDIM; ++d)
      {
        F(d) = kappa_s_block * (X(d) - x(d)) - eta_s_block * U[d];
      }
    return;
  } // block_tether_force_function

  // Tether (penalty) force function for the thin beam.
  static libMesh::BoundaryInfo *beam_boundary_info;
  static double                 kappa_s_beam = 1.0e6;
  static double                 eta_s_beam   = 0.0;
  void
  beam_tether_force_function(
    libMesh::VectorValue<double> &F,
    const libMesh::VectorValue<double> & /*n*/,
    const libMesh::VectorValue<double> & /*N*/,
    const libMesh::TensorValue<double> & /*FF*/,
    const libMesh::Point                           &x,
    const libMesh::Point                           &X,
    libMesh::Elem *const                            elem,
    const unsigned short                            side,
    const std::vector<const std::vector<double> *> &var_data,
    const std::vector<const std::vector<libMesh::VectorValue<double>> *>
      & /*grad_var_data*/,
    double /*time*/,
    void * /*ctx*/)
  {
    // Check to see if we are on the tethered boundaries.
    if (beam_boundary_info->has_boundary_id(elem, side, 1))
      {
        const std::vector<double> &U = *var_data[0];
        for (int d = 0; d < NDIM; ++d)
          {
            F(d) = kappa_s_beam * (X(d) - x(d)) - eta_s_beam * U[d];
          }
      }
    else
      {
        F.zero();
      }
    return;
  } // beam_tether_force_function

  // (Penalty) stress tensor function for the solid block.
  static double c1_s;
  void
  block_PK1_stress_function(
    libMesh::TensorValue<double>       &PP,
    const libMesh::TensorValue<double> &FF,
    const libMesh::Point & /*X*/,
    const libMesh::Point & /*s*/,
    libMesh::Elem *const /*elem*/,
    const std::vector<const std::vector<double> *> & /*var_data*/,
    const std::vector<const std::vector<libMesh::VectorValue<double>> *>
      & /*grad_var_data*/,
    double /*time*/,
    void * /*ctx*/)
  {
    PP = 2.0 * c1_s * (FF - tensor_inverse_transpose(FF, NDIM));
    return;
  } // block_PK1_stress_function

  // Stress tensor function for the thin beam.
  static double mu_s, lambda_s;
  void
  beam_PK1_dev_stress_function(
    libMesh::TensorValue<double>       &PP,
    const libMesh::TensorValue<double> &FF,
    const libMesh::Point & /*X*/,
    const libMesh::Point & /*s*/,
    libMesh::Elem *const /*elem*/,
    const std::vector<const std::vector<double> *> & /*var_data*/,
    const std::vector<const std::vector<libMesh::VectorValue<double>> *>
      & /*grad_var_data*/,
    double /*time*/,
    void * /*ctx*/)
  {
    static const libMesh::TensorValue<double> II(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    const libMesh::TensorValue<double> CC = FF.transpose() * FF;

    //  Modified Neo-Hookean Model
    const double                       J             = FF.det();
    const double                       J_cbrt_inv    = 1.0 / cbrt(J);
    const double                       J_cbrt_inv_sq = J_cbrt_inv * J_cbrt_inv;
    const double                       I1 = (FF.transpose() * FF).tr();
    const libMesh::TensorValue<double> FF_inv_trans =
      tensor_inverse_transpose(FF, NDIM);
    PP = mu_s * J_cbrt_inv_sq * (FF - (I1 / 3.0) * FF_inv_trans);

//  Unmodified St. Venant-Kirchhoff Model
#if 0
    const libMesh::TensorValue<double> EE = 0.5 * (CC - II);
    const libMesh::TensorValue<double> SS = lambda_s * EE.tr() * II + 2.0 * mu_s * EE;
    PP = FF * SS;
#endif

//  Modified St. Venant-Kirchhoff Model
#if 0
    const libMesh::TensorValue<double> CC_sq = CC * CC;
    const double J_c = CC.det();
    const double J_c_1_3 = pow(J_c, -1.0 / 3.0);
    const double I1 = CC.tr();
    const double I2 = 0.5 * (I1*I1 - CC_sq.tr());
    const double I1_bar = J_c_1_3 * I1;
    const double I2_bar = J_c_1_3 * J_c_1_3 * I2;
    const libMesh::TensorValue<double> CC_inv_trans = tensor_inverse_transpose(CC, NDIM);
    const libMesh::TensorValue<double> dW_dCC =
        (lambda_s / 4.0) * (I1_bar - 3.0) * (J_c_1_3 * II - (I1_bar / 3.0) * CC_inv_trans) +
        (mu_s / 2.0) *
            (J_c_1_3 * (J_c_1_3 * CC - II) + (1.0 / 3.0) * (I1_bar * (1.0 - I1_bar) + 2.0 * I2_bar) * CC_inv_trans);
    PP = 2.0 * FF * dW_dCC;
#endif
    return;
  } // beam_PK1_stress_function

  // Dilational stress tensor function for the thin beam.
  static double beta_s;
  void
  beam_PK1_dil_stress_function(
    libMesh::TensorValue<double>       &PP,
    const libMesh::TensorValue<double> &FF,
    const libMesh::Point & /*X*/,
    const libMesh::Point & /*s*/,
    libMesh::Elem *const /*elem*/,
    const std::vector<const std::vector<double> *> & /*var_data*/,
    const std::vector<const std::vector<libMesh::VectorValue<double>> *>
      & /*grad_var_data*/,
    double /*time*/,
    void * /*ctx*/)
  {
    double                       J = FF.det();
    libMesh::TensorValue<double> FF_inv_trans =
      tensor_inverse_transpose(FF, NDIM);
    PP = beta_s * J * log(J) * FF_inv_trans;
  }

} // namespace ModelData
using namespace ModelData;

// Function prototypes
static std::ofstream drag_stream, lift_stream, A_x_posn_stream, A_y_posn_stream;
void
postprocess_data(tbox::Pointer<PatchHierarchy<NDIM>>   patch_hierarchy,
                 tbox::Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
                 libMesh::MeshBase                    &beam_mesh,
                 libMesh::EquationSystems             *beam_equation_systems,
                 libMesh::MeshBase                    &block_mesh,
                 libMesh::EquationSystems             *block_equation_systems,
                 const int                             iteration_num,
                 const double                          loop_time,
                 const std::string                    &data_dump_dirname);

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
  // Initialize libMesh, PETSc, MPI, and SAMRAI.
  libMesh::LibMeshInit init(argc, argv);
  tbox::SAMRAI_MPI::setCommunicator(PETSC_COMM_WORLD);
  tbox::SAMRAI_MPI::setCallAbortInSerialInsteadOfExit();
  tbox::SAMRAIManager::startup();

  { // cleanup dynamically allocated objects prior to shutdown

    // Parse command line options, set some standard options from the input
    // file, initialize the restart database (if this is a restarted run),
    // and enable file logging.
    tbox::Pointer<IBTK::AppInitializer> app_initializer =
      new IBTK::AppInitializer(argc, argv, "IB.log");
    tbox::Pointer<Database> input_db = app_initializer->getInputDatabase();

    // Get various standard options set in the input file.
    const bool dump_viz_data     = app_initializer->dumpVizData();
    const int  viz_dump_interval = app_initializer->getVizDumpInterval();
    const bool uses_visit =
      dump_viz_data && app_initializer->getVisItDataWriter();
    const bool uses_exodus =
      dump_viz_data && !app_initializer->getExodusIIFilename().empty();
    const std::string block_exodus_filename =
      app_initializer->getExodusIIFilename("block");
    const std::string beam_exodus_filename =
      app_initializer->getExodusIIFilename("beam");

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
        Utilities::recursiveMkdir(postproc_data_dump_dirname);
      }

    const bool dump_timer_data     = app_initializer->dumpTimerData();
    const int  timer_dump_interval = app_initializer->getTimerDumpInterval();

    // Create a simple FE mesh.
    const double dx = input_db->getDouble("DX");
    const double ds = input_db->getDouble("MFAC") * dx;
    const int    second_order_block_mesh =
      (input_db->getString("block_elem_order") == "SECOND");
    const int second_order_beam_mesh =
      (input_db->getString("beam_elem_order") == "SECOND");
    const int selective_reduced_integration =
      input_db->getBool("SELECTIVE_REDUCED_INTEGRATION");

    libMesh::ReplicatedMesh block_mesh(init.comm(), NDIM);
    std::string  block_elem_type = input_db->getString("BLOCK_ELEM_TYPE");
    const double R               = 0.05;
    if (block_elem_type == "TRI3" || block_elem_type == "TRI6")
      {
#ifdef LIBMESH_HAVE_TRIANGLE
        const int num_circum_nodes = std::ceil(2.0 * M_PI * R / ds);
        for (int k = 0; k < num_circum_nodes; ++k)
          {
            const double theta = 2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(num_circum_nodes);
            block_mesh.add_point(
              libMesh::Point(R * cos(theta), R * sin(theta)));
          }
        TriangleInterface triangle(block_mesh);
        triangle.triangulation_type() = TriangleInterface::GENERATE_CONVEX_HULL;
        triangle.elem_type() =
          Utility::string_to_enum<ElemType>(block_elem_type);
        triangle.desired_area()            = sqrt(3.0) / 4.0 * ds * ds;
        triangle.insert_extra_points()     = true;
        triangle.smooth_after_generating() = true;
        triangle.triangulate();
#else
        TBOX_ERROR(
          "ERROR: libMesh appears to have been configured without support for "
          "Triangle,\n"
          << "       but Triangle is required for TRI3 or TRI6 elements.\n");
#endif
      }
    else
      {
        // NOTE: number of segments along boundary is 4*2^r.
        const double num_circum_segments = std::ceil(2.0 * M_PI * R / ds);
        const int    r                   = log2(0.25 * num_circum_segments);
        MeshTools::Generation::build_sphere(
          block_mesh, R, r, Utility::string_to_enum<ElemType>(block_elem_type));
      }
    for (libMesh::MeshBase::node_iterator n_it = block_mesh.nodes_begin();
         n_it != block_mesh.nodes_end();
         ++n_it)
      {
        Node &n = **n_it;
        n(0) += 0.2;
        n(1) += 0.2;
      }

    block_mesh.prepare_for_use();

    BoundaryMesh boundary_mesh(block_mesh.comm(),
                               block_mesh.mesh_dimension() - 1);
    block_mesh.boundary_info->sync(boundary_mesh);
    boundary_mesh.prepare_for_use();

    bool use_boundary_mesh =
      input_db->getBoolWithDefault("USE_BOUNDARY_MESH", false);
    pout << "use_boundary_mesh = " << use_boundary_mesh << "\n";

    libMesh::ReplicatedMesh beam_mesh(init.comm(), NDIM);
    std::string beam_elem_type = input_db->getString("BEAM_ELEM_TYPE");
    beam_mesh.read(input_db->getString("BEAM_MESH_FILENAME"), NULL);
    beam_mesh.prepare_for_use();

    beam_boundary_info = &beam_mesh.get_boundary_info();

    std::vector<libMesh::MeshBase *> meshes(2);
    meshes[0] = use_boundary_mesh ? &boundary_mesh : &block_mesh;
    meshes[1] = &beam_mesh;

    mu_s     = input_db->getDouble("MU_S");
    lambda_s = input_db->getDouble("LAMBDA_S");

    c1_s          = input_db->getDouble("C1_S");
    beta_s        = input_db->getDouble("BETA_S");
    kappa_s_block = input_db->getDouble("KAPPA_S_BLOCK");
    eta_s_block   = input_db->getDouble("ETA_S_BLOCK");
    kappa_s_beam  = input_db->getDouble("KAPPA_S_BEAM");
    eta_s_beam    = input_db->getDouble("ETA_S_BEAM");

    // Create major algorithm and data objects that comprise the
    // application.  These objects are configured from the input database
    // and, if this is a restarted run, from the restart database.
    tbox::Pointer<INSHierarchyIntegrator> navier_stokes_integrator;
    const std::string                     solver_type =
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
    tbox::Pointer<IBAMR::IBFEMethod> ib_method_ops =
      new IBAMR::IBFEMethod("IBFEMethod",
                            app_initializer->getComponentDatabase("IBFEMethod"),
                            meshes,
                            app_initializer
                              ->getComponentDatabase("GriddingAlgorithm")
                              ->getInteger("max_levels"),
                            /*register_for_restart*/ true,
                            app_initializer->getRestartDumpDirectory(),
                            app_initializer->getRestartRestoreNumber());
    tbox::Pointer<IBAMR::IBHierarchyIntegrator> time_integrator =
      new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                        app_initializer->getComponentDatabase(
                                          "IBHierarchyIntegrator"),
                                        ib_method_ops,
                                        navier_stokes_integrator);
    tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
      new CartesianGridGeometry<NDIM>("CartesianGeometry",
                                      app_initializer->getComponentDatabase(
                                        "CartesianGeometry"));
    tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy =
      new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
    tbox::Pointer<mesh::StandardTagAndInitialize<NDIM>> error_detector =
      new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                         time_integrator,
                                         app_initializer->getComponentDatabase(
                                           "StandardTagAndInitialize"));
    tbox::Pointer<mesh::BergerRigoutsos<NDIM>> box_generator =
      new BergerRigoutsos<NDIM>();
    tbox::Pointer<mesh::LoadBalancer<NDIM>> load_balancer =
      new LoadBalancer<NDIM>(
        "LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
    tbox::Pointer<mesh::GriddingAlgorithm<NDIM>> gridding_algorithm =
      new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                  app_initializer->getComponentDatabase(
                                    "GriddingAlgorithm"),
                                  error_detector,
                                  box_generator,
                                  load_balancer);

    // Configure the IBFE solver.
    std::string block_kernel_fcn =
      input_db->getStringWithDefault("BLOCK_KERNEL_FUNCTION",
                                     "PIECEWISE_LINEAR");
    IBTK::FEDataManager::InterpSpec block_interp_spec =
      ib_method_ops->getDefaultInterpSpec();
    block_interp_spec.kernel_fcn = block_kernel_fcn;
    ib_method_ops->setInterpSpec(block_interp_spec, 0);
    IBTK::FEDataManager::SpreadSpec block_spread_spec =
      ib_method_ops->getDefaultSpreadSpec();
    block_spread_spec.kernel_fcn = block_kernel_fcn;
    ib_method_ops->setSpreadSpec(block_spread_spec, 0);

    std::string beam_kernel_fcn =
      input_db->getStringWithDefault("BEAM_KERNEL_FUNCTION",
                                     "PIECEWISE_LINEAR");
    IBTK::FEDataManager::InterpSpec beam_interp_spec =
      ib_method_ops->getDefaultInterpSpec();
    beam_interp_spec.kernel_fcn = beam_kernel_fcn;
    ib_method_ops->setInterpSpec(beam_interp_spec, 1);
    IBTK::FEDataManager::SpreadSpec beam_spread_spec =
      ib_method_ops->getDefaultSpreadSpec();
    beam_spread_spec.kernel_fcn = beam_kernel_fcn;
    ib_method_ops->setSpreadSpec(beam_spread_spec, 1);

    ib_method_ops->initializeFEEquationSystems();
    std::vector<int> vars(NDIM);
    for (unsigned int d = 0; d < NDIM; ++d)
      vars[d] = d;
    std::vector<SystemData> sys_data(
      1, SystemData(IBAMR::IBFEMethod::VELOCITY_SYSTEM_NAME, vars));
    if (use_boundary_mesh)
      {
        IBAMR::IBFEMethod::LagBodyForceFcnData block_tether_force_data(
          block_tether_force_function, sys_data);
        ib_method_ops->registerLagBodyForceFunction(block_tether_force_data, 0);
      }
    else
      {
        IBAMR::IBFEMethod::LagBodyForceFcnData block_tether_force_data(
          block_tether_force_function, sys_data);
        IBAMR::IBFEMethod::PK1StressFcnData block_PK1_stress_data(
          block_PK1_stress_function);
        block_PK1_stress_data.quad_order =
          Utility::string_to_enum<libMesh::Order>(
            input_db->getStringWithDefault("PK1_QUAD_ORDER",
                                           second_order_block_mesh ? "FIFTH" :
                                                                     "THIRD"));
        ib_method_ops->registerLagBodyForceFunction(block_tether_force_data, 0);
        ib_method_ops->registerPK1StressFunction(block_PK1_stress_data, 0);
        if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
          {
            ib_method_ops->registerStressNormalizationPart(0);
          }
      }
    IBAMR::IBFEMethod::LagSurfaceForceFcnData beam_tether_force_data(
      beam_tether_force_function, sys_data);
    IBAMR::IBFEMethod::PK1StressFcnData beam_PK1_dev_stress_data(
      beam_PK1_dev_stress_function);
    beam_PK1_dev_stress_data.quad_order =
      Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
        "PK1_QUAD_ORDER", second_order_beam_mesh ? "FIFTH" : "THIRD"));
    IBAMR::IBFEMethod::PK1StressFcnData beam_PK1_dil_stress_data(
      beam_PK1_dil_stress_function);
    beam_PK1_dil_stress_data.quad_order =
      Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
        "PK1_QUAD_ORDER",
        second_order_beam_mesh ?
          (selective_reduced_integration ? "THIRD" : "FIFTH") :
          "THIRD"));
    ib_method_ops->registerLagSurfaceForceFunction(beam_tether_force_data, 1);
    ib_method_ops->registerPK1StressFunction(beam_PK1_dev_stress_data, 1);
    ib_method_ops->registerPK1StressFunction(beam_PK1_dil_stress_data, 1);
    if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
      {
        ib_method_ops->registerStressNormalizationPart(1);
      }

    libMesh::EquationSystems *block_equation_systems =
      ib_method_ops->getFEDataManager(0)->getEquationSystems();
    libMesh::EquationSystems *beam_equation_systems =
      ib_method_ops->getFEDataManager(1)->getEquationSystems();

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
    std::unique_ptr<libMesh::ExodusII_IO> block_exodus_io(
      uses_exodus ? new libMesh::ExodusII_IO(*meshes[0]) : NULL);
    std::unique_ptr<libMesh::ExodusII_IO> beam_exodus_io(
      uses_exodus ? new libMesh::ExodusII_IO(*meshes[1]) : NULL);

    const bool from_restart =
      tbox::RestartManager::getManager()->isFromRestart();
    // if (block_exodus_io) block_exodus_io->append(from_restart);
    // if (beam_exodus_io) beam_exodus_io->append(from_restart);

    // Initialize hierarchy configuration and data on all patches.
    ib_method_ops->initializeFEData();
    time_integrator->initializePatchHierarchy(patch_hierarchy,
                                              gridding_algorithm);

    // Deallocate initialization objects.
    app_initializer.setNull();

    // Print the input database contents to the log file.
    plog << "Input database:\n";
    input_db->printClassData(plog);

    // Write out initial visualization data.
    int    iteration_num = time_integrator->getIntegratorStep();
    double loop_time     = time_integrator->getIntegratorTime();
    if (dump_viz_data)
      {
        pout << "\n\nWriting visualization files...\n\n";
        if (uses_visit)
          {
            time_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy,
                                             iteration_num,
                                             loop_time);
          }
        if (uses_exodus)
          {
            block_exodus_io->write_timestep(block_exodus_filename,
                                            *block_equation_systems,
                                            iteration_num / viz_dump_interval +
                                              1,
                                            loop_time);
            beam_exodus_io->write_timestep(beam_exodus_filename,
                                           *beam_equation_systems,
                                           iteration_num / viz_dump_interval +
                                             1,
                                           loop_time);
          }
      }

    // Open streams to save lift and drag coefficients.
    if (tbox::SAMRAI_MPI::getRank() == 0)
      {
        drag_stream.open("C_D.curve", ios_base::out | ios_base::trunc);
        lift_stream.open("C_L.curve", ios_base::out | ios_base::trunc);
        A_x_posn_stream.open("A_x.curve", ios_base::out | ios_base::trunc);
        A_y_posn_stream.open("A_y.curve", ios_base::out | ios_base::trunc);
      }

    // Main time step loop.
    double loop_time_end = time_integrator->getEndTime();
    double dt            = 0.0;
    while (!tbox::MathUtilities<double>::equalEps(loop_time, loop_time_end) &&
           time_integrator->stepsRemaining())
      {
        iteration_num = time_integrator->getIntegratorStep();
        loop_time     = time_integrator->getIntegratorTime();

        pout << "\n";
        pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
        pout << "At beginning of timestep # " << iteration_num << "\n";
        pout << "Simulation time is " << loop_time << "\n";

        dt = time_integrator->getMaximumTimeStepSize();
        time_integrator->advanceHierarchy(dt);
        loop_time += dt;

        pout << "\n";
        pout << "At end       of timestep # " << iteration_num << "\n";
        pout << "Simulation time is " << loop_time << "\n";
        pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
        pout << "\n";

        // At specified intervals, write visualization and restart files,
        // print out timer data, and store hierarchy data for post
        // processing.
        iteration_num += 1;
        const bool last_step = !time_integrator->stepsRemaining();
        if (dump_viz_data &&
            (iteration_num % viz_dump_interval == 0 || last_step))
          {
            pout << "\nWriting visualization files...\n\n";
            if (uses_visit)
              {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy,
                                                 iteration_num,
                                                 loop_time);
              }
            if (uses_exodus)
              {
                block_exodus_io->write_timestep(block_exodus_filename,
                                                *block_equation_systems,
                                                iteration_num /
                                                    viz_dump_interval +
                                                  1,
                                                loop_time);
                beam_exodus_io->write_timestep(beam_exodus_filename,
                                               *beam_equation_systems,
                                               iteration_num /
                                                   viz_dump_interval +
                                                 1,
                                               loop_time);
              }
          }
        if (dump_restart_data &&
            (iteration_num % restart_dump_interval == 0 || last_step))
          {
            pout << "\nWriting restart files...\n\n";
            tbox::RestartManager::getManager()->writeRestartFile(
              restart_dump_dirname, iteration_num);
            ib_method_ops->writeFEDataToRestartFile(restart_dump_dirname,
                                                    iteration_num);
          }
        if (dump_timer_data &&
            (iteration_num % timer_dump_interval == 0 || last_step))
          {
            pout << "\nWriting timer data...\n\n";
            tbox::TimerManager::getManager()->print(plog);
          }
        if (dump_postproc_data &&
            (iteration_num % postproc_data_dump_interval == 0 || last_step))
          {
            pout << "\nWriting state data...\n\n";
            postprocess_data(patch_hierarchy,
                             navier_stokes_integrator,
                             beam_mesh,
                             beam_equation_systems,
                             *meshes[0],
                             block_equation_systems,
                             iteration_num,
                             loop_time,
                             postproc_data_dump_dirname);
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

  tbox::SAMRAIManager::shutdown();
}

void
postprocess_data(
  tbox::Pointer<PatchHierarchy<NDIM>> /*patch_hierarchy*/,
  tbox::Pointer<INSHierarchyIntegrator> /*navier_stokes_integrator*/,
  libMesh::MeshBase        &beam_mesh,
  libMesh::EquationSystems *beam_equation_systems,
  libMesh::MeshBase        &block_mesh,
  libMesh::EquationSystems *block_equation_systems,
  const int /*iteration_num*/,
  const double loop_time,
  const std::string & /*data_dump_dirname*/)
{
  double F_integral[NDIM];
  for (unsigned int d = 0; d < NDIM; ++d)
    F_integral[d] = 0.0;
  libMesh::MeshBase        *mesh[2]             = {&beam_mesh, &block_mesh};
  libMesh::EquationSystems *equation_systems[2] = {beam_equation_systems,
                                                   block_equation_systems};
  for (unsigned int k = 0; k < 2; ++k)
    {
      const unsigned int dim = mesh[k]->mesh_dimension();
      libMesh::System   &F_system =
        equation_systems[k]->get_system<libMesh::System>(
          IBAMR::IBFEMethod::FORCE_SYSTEM_NAME);
      libMesh::NumericVector<double> *F_vec = F_system.solution.get();
      libMesh::NumericVector<double> *F_ghost_vec =
        F_system.current_local_solution.get();
      F_vec->localize(*F_ghost_vec);
      libMesh::System &x_system =
        equation_systems[k]->get_system<libMesh::System>(
          IBAMR::IBFEMethod::COORDS_SYSTEM_NAME);
      libMesh::System &U_system =
        equation_systems[k]->get_system<libMesh::System>(
          IBAMR::IBFEMethod::VELOCITY_SYSTEM_NAME);
      libMesh::NumericVector<double> *x_vec = x_system.solution.get();
      libMesh::NumericVector<double> *x_ghost_vec =
        x_system.current_local_solution.get();
      x_vec->localize(*x_ghost_vec);
      libMesh::NumericVector<double> *U_vec = U_system.solution.get();
      libMesh::NumericVector<double> *U_ghost_vec =
        U_system.current_local_solution.get();
      U_vec->localize(*U_ghost_vec);

      libMesh::DofMap                       &F_dof_map = F_system.get_dof_map();
      std::vector<std::vector<unsigned int>> F_dof_indices(NDIM);

      std::unique_ptr<FEBase> fe(
        FEBase::build(dim, F_dof_map.variable_type(0)));
      std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, SEVENTH);
      fe->attach_quadrature_rule(qrule.get());
      const std::vector<double>              &JxW     = fe->get_JxW();
      const std::vector<libMesh::Point>      &q_point = fe->get_xyz();
      const std::vector<std::vector<double>> &phi     = fe->get_phi();
      const std::vector<std::vector<libMesh::VectorValue<double>>> &dphi =
        fe->get_dphi();

      std::vector<double>                      U_qp_vec(NDIM);
      std::vector<const std::vector<double> *> var_data(1);
      var_data[0] = &U_qp_vec;
      std::vector<const std::vector<libMesh::VectorValue<double>> *>
            grad_var_data;
      void *force_fcn_ctx = NULL;

      libMesh::TensorValue<double>  FF_qp;
      boost::multi_array<double, 2> F_node, x_node, U_node;
      libMesh::VectorValue<double>  F_qp, U_qp, x_qp;

      const libMesh::MeshBase::const_element_iterator el_begin =
        mesh[k]->active_local_elements_begin();
      const libMesh::MeshBase::const_element_iterator el_end =
        mesh[k]->active_local_elements_end();
      for (libMesh::MeshBase::const_element_iterator el_it = el_begin;
           el_it != el_end;
           ++el_it)
        {
          libMesh::Elem *const elem = *el_it;
          fe->reinit(elem);
          for (unsigned int d = 0; d < NDIM; ++d)
            {
              F_dof_map.dof_indices(elem, F_dof_indices[d], d);
            }
          get_values_for_interpolation(F_node, *F_ghost_vec, F_dof_indices);
          get_values_for_interpolation(x_node, *x_ghost_vec, F_dof_indices);
          get_values_for_interpolation(U_node, *U_ghost_vec, F_dof_indices);

          const int n_qp    = qrule->n_points();
          const int n_basis = static_cast<int>(F_dof_indices[0].size());
          for (int qp = 0; qp < n_qp; ++qp)
            {
              if (k == 0)
                {
                  for (int k = 0; k < n_basis; ++k)
                    {
                      for (int d = 0; d < NDIM; ++d)
                        {
                          F_integral[d] += F_node[k][d] * phi[k][qp] * JxW[qp];
                        }
                    }
                }
              else
                {
                  interpolate(x_qp, qp, x_node, phi);
                  jacobian(FF_qp, qp, x_node, dphi);
                  interpolate(U_qp, qp, U_node, phi);
                  for (unsigned int d = 0; d < NDIM; ++d)
                    {
                      U_qp_vec[d] = U_qp(d);
                    }
                  block_tether_force_function(F_qp,
                                              FF_qp,
                                              x_qp,
                                              q_point[qp],
                                              elem,
                                              var_data,
                                              grad_var_data,
                                              loop_time,
                                              force_fcn_ctx);
                  for (int d = 0; d < NDIM; ++d)
                    {
                      F_integral[d] += F_qp(d) * JxW[qp];
                    }
                }
            }
        }
    }
  tbox::SAMRAI_MPI::sumReduction(F_integral, NDIM);
  if (tbox::SAMRAI_MPI::getRank() == 0)
    {
      drag_stream.precision(12);
      drag_stream.setf(std::ios::fixed, std::ios::floatfield);
      drag_stream << loop_time << " " << -F_integral[0] << endl;
      lift_stream.precision(12);
      lift_stream.setf(std::ios::fixed, std::ios::floatfield);
      lift_stream << loop_time << " " << -F_integral[1] << endl;
    }

  libMesh::System &X_system =
    beam_equation_systems->get_system<libMesh::System>(
      IBAMR::IBFEMethod::COORDS_SYSTEM_NAME);
  libMesh::NumericVector<double> *X_vec = X_system.solution.get();
  std::unique_ptr<libMesh::NumericVector<double>> X_serial_vec =
    libMesh::NumericVector<double>::build(X_vec->comm());
  X_serial_vec->init(X_vec->size(), true, SERIAL);
  X_vec->localize(*X_serial_vec);
  libMesh::DofMap          &X_dof_map = X_system.get_dof_map();
  std::vector<unsigned int> vars(2);
  vars[0] = 0;
  vars[1] = 1;
  libMesh::MeshFunction X_fcn(*beam_equation_systems,
                              *X_serial_vec,
                              X_dof_map,
                              vars);
  X_fcn.init();
  libMesh::DenseVector<double> X_A(2);
  X_fcn(libMesh::Point(0.6, 0.2, 0), 0.0, X_A);
  if (tbox::SAMRAI_MPI::getRank() == 0)
    {
      A_x_posn_stream.precision(12);
      A_x_posn_stream.setf(std::ios::fixed, std::ios::floatfield);
      A_x_posn_stream << loop_time << " " << X_A(0) << endl;
      A_y_posn_stream.precision(12);
      A_y_posn_stream.setf(std::ios::fixed, std::ios::floatfield);
      A_y_posn_stream << loop_time << " " << X_A(1) << endl;
    }
  return;
} // postprocess_data
