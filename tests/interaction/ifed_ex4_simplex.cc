#include <fiddle/interaction/ifed_method.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <fstream>
#include <vector>

#include "../tests.h"

using namespace dealii;
using namespace SAMRAI;

// Putting it all together - fdl::IFEDMethod version of IBFE/explicit/ex4

template <int dim, int spacedim = dim>
class DeviatoricStress : public fdl::ForceContribution<dim, spacedim>
{
public:
  // TODO - find a better name for c1
  DeviatoricStress(const Quadrature<dim> &quadrature, const double c1)
    : fdl::ForceContribution<dim, spacedim>(quadrature)
    , c1(c1)
  {}

  virtual bool
  is_stress() const override
  {
    return true;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::MechanicsUpdateFlags::update_FF;
  }

  virtual void
  compute_stress(
    const double /*time*/,
    const fdl::MechanicsValues<dim, spacedim> &me_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      & /*cell*/,
    ArrayView<Tensor<2, spacedim, double>> &stresses) const override
  {
    const std::vector<Tensor<2, spacedim>> &FF = me_values.get_FF();
    Assert(FF.size() == stresses.size(), ExcMessage("sizes should match"));
    for (unsigned int qp_n = 0; qp_n < FF.size(); ++qp_n)
      stresses[qp_n] = 2.0 * c1 * FF[qp_n];
  }


private:
  double c1;
};

template <int dim, int spacedim = dim>
class DilationalStress : public fdl::ForceContribution<dim, spacedim>
{
public:
  // TODO - find better names for p0 and beta
  DilationalStress(const Quadrature<dim> &quadrature,
                   const double           p0,
                   const double           beta)
    : fdl::ForceContribution<dim, spacedim>(quadrature)
    , p0(p0)
    , beta(beta)
  {}

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::MechanicsUpdateFlags::update_FF_inv_T |
           fdl::MechanicsUpdateFlags::update_det_FF;
  }

  virtual bool
  is_stress() const override
  {
    return true;
  }


  virtual void
  compute_stress(
    const double /*time*/,
    const fdl::MechanicsValues<dim, spacedim> &me_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      & /*cell*/,
    ArrayView<Tensor<2, spacedim, double>> &stresses) const override
  {
    const std::vector<double>              &det_FF   = me_values.get_det_FF();
    const std::vector<Tensor<2, spacedim>> &FF_inv_T = me_values.get_FF_inv_T();
    Assert(det_FF.size() == stresses.size(), ExcMessage("sizes should match"));
    for (unsigned int qp_n = 0; qp_n < det_FF.size(); ++qp_n)
      stresses[qp_n] =
        2.0 * (-p0 + beta * std::log(det_FF[qp_n])) * FF_inv_T[qp_n];
  }

private:
  double p0;
  double beta;
};

template <int dim, int spacedim = dim>
void
test(tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  // suppress warnings caused by using a refinement ratio of 4 and not
  // setting up coarsening correctly
  SAMRAI::tbox::Logger::getInstance()->setWarning(false);

  auto       input_db = app_initializer->getInputDatabase();
  auto       test_db  = input_db->getDatabase("test");
  const auto mpi_comm = MPI_COMM_WORLD;

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(
    mpi_comm, {}, test_db->getBoolWithDefault("use_artificial_cells", true));

  Point<dim> center;
  center[0]       = 0.6;
  center[1]       = 0.5;
  center[dim - 1] = 0.5; // works in 2D and 3D
#if 0
  Triangulation<dim, spacedim> native_hex_tria;
  GridGenerator::hyper_ball(native_hex_tria, center, 0.2);
  // GridGenerator::subdivided_hyper_cube(native_hex_tria, 1, 0.4, 0.6);
  GridGenerator::convert_hypercube_to_simplex_mesh(native_hex_tria,
                                                   native_tria);

  for (const auto id : native_hex_tria.get_manifold_ids())
    if (id != numbers::flat_manifold_id)
      native_tria.set_manifold(id, native_hex_tria.get_manifold(id));

  // Multiply by 2.0 at the end to match what libMesh/IBFE/explicit/ex4 does
  const double target_element_size =
    input_db->getDouble("MFAC") * input_db->getDouble("DX") * 2.0;
  while (GridTools::maximal_cell_diameter(native_tria) > target_element_size)
    native_tria.refine_global(1);
  // TODO - this is a workaround for an unimplemented feature in MappingFE.
  // MappingFE needs to learn how to do cellwise TFI.
  native_tria.reset_all_manifolds();
  native_tria.set_all_manifold_ids(numbers::flat_manifold_id);
#else
  GridIn<dim, spacedim> grid_in(native_tria);
  const auto            exodusii_data = grid_in.read_exodusii("output.ex2");
  GridTools::shift(center, native_tria);
#endif

  tbox::pout << "Number of elements = " << native_tria.n_active_cells() << '\n';

  // fiddle stuff:
  FESystem<dim> fe(FE_SimplexP<dim>(input_db->getInteger("fe_degree")), dim);
  std::vector<fdl::Part<dim>> parts;

  QWitherdenVincentSimplex<dim> dev_quad(
    input_db->getIntegerWithDefault("pk1_dev_n_points_1d", 2));
  QWitherdenVincentSimplex<dim> dil_quad(
    input_db->getIntegerWithDefault("pk1_dil_n_points_1d", 1));
  std::vector<std::unique_ptr<fdl::ForceContribution<dim, spacedim>>> forces;
  forces.emplace_back(new DeviatoricStress<dim, spacedim>(
    dev_quad, input_db->getDoubleWithDefault("c1", 0.05)));
  forces.emplace_back(new DilationalStress<dim, spacedim>(
    dil_quad,
    input_db->getDoubleWithDefault("p0", 0.0),
    input_db->getDoubleWithDefault("beta", 0.0)));

  parts.emplace_back(native_tria, fe, std::move(forces));
  tbox::Pointer<fdl::IFEDMethod<spacedim>> ib_method_ops =
    new fdl::IFEDMethod<dim>("ifed_method",
                             input_db->getDatabase("IFEDMethod"),
                             std::move(parts));

  // Create major algorithm and data objects that comprise the
  // application. These objects are configured from the input database
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

  // Now that everything is set up we can register one more plotting quantity
  visit_data_writer->registerPlotQuantity(
    "quadraturepoints",
    "SCALAR",
    ib_method_ops->get_lagrangian_workload_current_index());

  // Write out initial visualization data.
  int    iteration_num = time_integrator->getIntegratorStep();
  double loop_time     = time_integrator->getIntegratorTime();
  time_integrator->setupPlotData();
  visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
  {
    const auto  &part = ib_method_ops->get_part(0);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(part.get_dof_handler());
    data_out.add_data_vector(part.get_velocity(), "U");

    MappingFEField<dim, spacedim, LinearAlgebra::distributed::Vector<double>>
      position_mapping(part.get_dof_handler(), part.get_position());
    data_out.build_patches(position_mapping);
    data_out.write_vtu_with_pvtu_record(app_initializer->getVizDumpDirectory() +
                                          "/",
                                        "solution",
                                        iteration_num,
                                        mpi_comm,
                                        8);
  }

  std::ofstream volume_stream;
  if (IBTK::IBTK_MPI::getRank() == 0)
    {
      volume_stream.open("volume.curve",
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

      iteration_num += 1;
      const bool last_step = !time_integrator->stepsRemaining();
      if (last_step ||
          iteration_num % app_initializer->getVizDumpInterval() == 0)
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
            position_mapping(part.get_dof_handler(), part.get_position());
          data_out.build_patches(position_mapping);
          data_out.write_vtu_with_pvtu_record(
            app_initializer->getVizDumpDirectory() + "/",
            "solution",
            iteration_num,
            mpi_comm,
            8);
        }

      if (app_initializer->dumpRestartData() &&
          (iteration_num % app_initializer->getRestartDumpInterval() == 0))
        {
          tbox::pout << "\nWriting restart files...\n\n";
          tbox::RestartManager::getManager()->writeRestartFile(
            app_initializer->getRestartDumpDirectory(), iteration_num);
        }

      if (app_initializer->dumpTimerData() &&
          (iteration_num % app_initializer->getTimerDumpInterval() == 0 ||
           last_step))
        {
          tbox::pout << "\nWriting timer data...\n\n";
          tbox::TimerManager::getManager()->print(tbox::plog);
        }

      // Save structure volume:
      {
        const auto &part =
          dynamic_cast<fdl::IFEDMethod<NDIM> &>(*ib_method_ops).get_part(0);
        MappingFEField<dim,
                       spacedim,
                       LinearAlgebra::distributed::Vector<double>>
          position_mapping(part.get_dof_handler(), part.get_position());
        const double volume = GridTools::volume(native_tria, position_mapping);
        if (IBTK::IBTK_MPI::getRank() == 0)
          {
            volume_stream.precision(12);
            volume_stream.setf(std::ios::fixed, std::ios::floatfield);
            volume_stream << loop_time << " " << volume << std::endl;
          }
      }
    }

  if (test_db->getBoolWithDefault("log_ends_of_fe_vectors", false))
    {
      const auto &part =
        dynamic_cast<fdl::IFEDMethod<NDIM> &>(*ib_method_ops).get_part(0);
      const auto &position = part.get_position();
      const auto &velocity = part.get_velocity();

      std::ostringstream out;
      out << std::setprecision(16);
      out << "rank = " << Utilities::MPI::this_mpi_process(mpi_comm) << '\n';
      out << "position:\n";
      const long a0 = 0;
      const long a1 = std::min<long>(position.locally_owned_size(), 5);
      const long b0 =
        std::max<long>(long(position.locally_owned_size()) - 5, 0);
      const long b1 = position.locally_owned_size();
      for (long i = a0; i < a1; ++i)
        out << position.local_element(i) << '\n';
      out << "...\n";
      for (long i = b0; i < b1; ++i)
        out << position.local_element(i) << '\n';
      out << '\n';
      out << "velocity:\n";
      for (long i = a0; i < a1; ++i)
        out << velocity.local_element(i) << '\n';
      out << "...\n";
      for (long i = b0; i < b1; ++i)
        out << velocity.local_element(i) << '\n';
      out << '\n';

      print_strings_on_0(out.str(), mpi_comm, tbox::pout);
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

  tbox::TimerManager::createManager(nullptr);
  test<NDIM>(app_initializer);
}
