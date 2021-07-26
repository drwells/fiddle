#include <fiddle/base/samrai_utilities.h>
#include <fiddle/base/utilities.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/grid_utilities.h>

#include <fiddle/interaction/elemental_interaction.h>
#include <fiddle/interaction/ifed_method.h>
#include <fiddle/interaction/interaction_utilities.h>

#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <ibamr/IBHierarchyIntegrator.h>
#include <ibamr/ibamr_utilities.h>

#include <ibtk/IBTK_MPI.h>

#include <CellVariable.h>
#include <HierarchyDataOpsManager.h>
#include <IntVector.h>
#include <VariableDatabase.h>
#include <tbox/RestartManager.h>
#include <tbox/TimerManager.h>

#include <deque>

namespace
{
  using namespace SAMRAI;
  static tbox::Timer *t_preprocess_integrate_data;
  static tbox::Timer *t_postprocess_integrate_data;
  static tbox::Timer *t_interpolate_velocity;
  static tbox::Timer *t_compute_lagrangian_force;
  static tbox::Timer *t_spread_force;
  static tbox::Timer *t_compute_lagrangian_fluid_source;
  static tbox::Timer *t_spread_fluid_source;
  static tbox::Timer *t_add_workload_estimate;
  static tbox::Timer *t_begin_data_redistribution;
  static tbox::Timer *t_end_data_redistribution;
  static tbox::Timer *t_apply_gradient_detector;
  static tbox::Timer *t_reinit_interactions_bboxes;
  static tbox::Timer *t_reinit_interactions_edges;
  static tbox::Timer *t_reinit_interactions_objects;
} // namespace

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  //
  // Initialization
  //

  template <int dim, int spacedim>
  IFEDMethod<dim, spacedim>::IFEDMethod(
    const std::string &                object_name,
    tbox::Pointer<tbox::Database>      input_input_db,
    std::vector<Part<dim, spacedim>> &&input_parts,
    const bool                         register_for_restart)
    : object_name(object_name)
    , register_for_restart(register_for_restart)
    , input_db(copy_database(input_input_db))
    , started_time_integration(false)
    , current_time(std::numeric_limits<double>::signaling_NaN())
    , half_time(std::numeric_limits<double>::signaling_NaN())
    , new_time(std::numeric_limits<double>::signaling_NaN())
    , parts(std::move(input_parts))
    , part_vectors(this->parts)
    , secondary_hierarchy(object_name + "::secondary_hierarchy",
                          input_db->getDatabase("GriddingAlgorithm"),
                          input_db->getDatabase("LoadBalancer"))
  {
    // IBFEMethod uses this value - lower values aren't guaranteed to work. If
    // dx = dX then we can use a lower density.
    const double density =
      input_db->getDoubleWithDefault("IB_point_density", 2.0);
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const unsigned int n_points_1D =
          parts[part_n].get_dof_handler().get_fe().tensor_degree() + 1;
        interactions.emplace_back(
          new ElementalInteraction<dim, spacedim>(n_points_1D, density));
      }

    AssertThrow(input_db->keyExists("IB_kernel"),
                ExcMessage(
                  "The IB kernel should be set in the input database."));
    // values in SAMRAI databases are always arrays, possibly of length 1
    const int n_ib_kernels = input_db->getArraySize("IB_kernel");
    AssertThrow(n_ib_kernels == 1 ||
                  n_ib_kernels == static_cast<int>(parts.size()),
                ExcMessage("The number of specified IB kernels should either "
                           "be 1 or equal the number of parts."));
    ib_kernels.resize(n_ib_kernels);
    input_db->getStringArray("IB_kernel",
                             ib_kernels.data(),
                             static_cast<int>(ib_kernels.size()));

    auto set_timer = [&](const char *name) {
      return tbox::TimerManager::getManager()->getTimer(name);
    };

    t_preprocess_integrate_data =
      set_timer("fdl::IFEDMethod::preprocessIntegrateData()");
    t_postprocess_integrate_data =
      set_timer("fdl::IFEDMethod::postprocessIntegrateData()");
    t_interpolate_velocity =
      set_timer("fdl::IFEDMethod::interpolateVelocity()");
    t_compute_lagrangian_force =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()");
    t_spread_force = set_timer("fdl::IFEDMethod::spreadForce()");
    t_compute_lagrangian_fluid_source =
      set_timer("fdl::IFEDMethod::computeLagrangianFluidSource()");
    t_spread_fluid_source = set_timer("fdl::IFEDMethod::spreadFluidSource()");
    t_add_workload_estimate =
      set_timer("fdl::IFEDMethod::addWorkloadEstimate()");
    t_begin_data_redistribution =
      set_timer("fdl::IFEDMethod::beginDataRedistribution()");
    t_end_data_redistribution =
      set_timer("fdl::IFEDMethod::endDataRedistribution()");
    t_apply_gradient_detector =
      set_timer("fdl::IFEDMethod::applyGradientDetector()");
    t_reinit_interactions_bboxes =
      set_timer("fdl::IFEDMethod::reinit_interactions()[bboxes]");
    t_reinit_interactions_edges =
      set_timer("fdl::IFEDMethod::reinit_interactions()[edges]");
    t_reinit_interactions_objects =
      set_timer("fdl::IFEDMethod::reinit_interactions()[objects]");

    if (register_for_restart)
      {
        auto *restart_manager = tbox::RestartManager::getManager();
        restart_manager->registerRestartItem(object_name, this);
        if (restart_manager->isFromRestart())
          {
            auto restart_db = restart_manager->getRootDatabase();
            if (restart_db->isDatabase(object_name))
              {
                auto db = restart_db->getDatabase(object_name);
                for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
                  {
                    const std::string key = "part_" + std::to_string(part_n);
                    AssertThrow(db->keyExists(key),
                                ExcMessage("Couldn't find key " + key +
                                           " in the restart database"));
                    const std::string  serialization = load_binary(key, db);
                    std::istringstream in_str(serialization);
                    boost::archive::binary_iarchive iarchive(in_str);
                    parts[part_n].load(iarchive, 0);
                  }
              }
            else
              {
                AssertThrow(false,
                            ExcMessage(
                              "The restart database does not contain key " +
                              object_name));
              }
          }
      }
  }

  template <int dim, int spacedim>
  IFEDMethod<dim, spacedim>::~IFEDMethod()
  {
    if (register_for_restart)
      {
        tbox::RestartManager::getManager()->unregisterRestartItem(object_name);
      }
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::initializePatchHierarchy(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/,
    int /*u_data_index*/,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      & /*u_synch_scheds*/,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      & /*u_ghost_fill_scheds*/,
    int /*integrator_step*/,
    double /*init_data_time*/,
    bool /*initial_time*/)
  {
    primary_hierarchy = hierarchy;

    primary_eulerian_data_cache = std::make_shared<IBTK::SAMRAIDataCache>();
    primary_eulerian_data_cache->setPatchHierarchy(hierarchy);
    primary_eulerian_data_cache->resetLevels(0,
                                             hierarchy->getFinestLevelNumber());

    secondary_hierarchy.reinit(primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy);

    reinit_interactions();
  }

  //
  // FSI
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::interpolateVelocity(
    int u_data_index,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      &u_synch_scheds,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      &    u_ghost_fill_scheds,
    double data_time)
  {
    IBAMR_TIMER_START(t_interpolate_velocity);
    (void)u_synch_scheds;
    (void)u_ghost_fill_scheds;

    // Update the secondary hierarchy:
    secondary_hierarchy
      .getPrimaryToSecondarySchedule(primary_hierarchy->getFinestLevelNumber(),
                                     u_data_index,
                                     u_data_index,
                                     d_ib_solver->getVelocityPhysBdryOp())
      .fillData(data_time);

    std::vector<std::unique_ptr<TransactionBase>> transactions;
    // we emplace_back so use a deque to keep pointers valid
    std::deque<LinearAlgebra::distributed::Vector<double>> rhs_vecs;

    // start:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const Part<dim, spacedim> &part = parts[part_n];
        rhs_vecs.emplace_back(part.get_partitioner());
        transactions.emplace_back(
          interactions[part_n]->compute_projection_rhs_start(
            ib_kernels[part_n],
            u_data_index,
            part.get_dof_handler(),
            part_vectors.get_position(part_n, data_time),
            part.get_dof_handler(),
            part.get_mapping(),
            rhs_vecs[part_n]));
      }

    // Compute:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      transactions[part_n] =
        interactions[part_n]->compute_projection_rhs_intermediate(
          std::move(transactions[part_n]));

    // Collect:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      interactions[part_n]->compute_projection_rhs_finish(
        std::move(transactions[part_n]));

    // Project:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        SolverControl control(1000, 1e-14 * rhs_vecs[part_n].l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
        LinearAlgebra::distributed::Vector<double>           velocity(
          parts[part_n].get_partitioner());
        // TODO - implement initial guess stuff here
        cg.solve(parts[part_n].get_mass_operator(),
                 velocity,
                 rhs_vecs[part_n],
                 parts[part_n].get_mass_preconditioner());
        part_vectors.set_velocity(part_n, data_time, std::move(velocity));
      }
    IBAMR_TIMER_STOP(t_interpolate_velocity);
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::spreadForce(
    int                               f_data_index,
    IBTK::RobinPhysBdryPatchStrategy *f_phys_bdry_op,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      & /*f_prolongation_scheds*/,
    double data_time)
  {
    IBAMR_TIMER_START(t_spread_force);
    const int level_number = primary_hierarchy->getFinestLevelNumber();

    std::shared_ptr<IBTK::SAMRAIDataCache> data_cache =
      secondary_hierarchy.getSAMRAIDataCache();
    auto       hierarchy = secondary_hierarchy.getSecondaryHierarchy();
    const auto f_scratch_data_index =
      data_cache->getCachedPatchDataIndex(f_data_index);
    fill_all(hierarchy, f_scratch_data_index, level_number, level_number, 0.0);

    // start:
    std::vector<std::unique_ptr<TransactionBase>> transactions;
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const Part<dim, spacedim> &part = parts[part_n];
        transactions.emplace_back(interactions[part_n]->compute_spread_start(
          ib_kernels[part_n],
          f_scratch_data_index,
          part_vectors.get_position(part_n, data_time),
          part.get_dof_handler(),
          part.get_mapping(),
          part.get_dof_handler(),
          part_vectors.get_force(part_n, data_time)));
      }

    // Compute:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      transactions[part_n] = interactions[part_n]->compute_spread_intermediate(
        std::move(transactions[part_n]));

    // Collect:
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      interactions[part_n]->compute_spread_finish(
        std::move(transactions[part_n]));

    // Deal with force values spread outside the physical domain. Since these
    // are spread into ghost regions that don't correspond to actual degrees
    // of freedom they are ignored by the accumulation step - we have to
    // handle this before we do that.
    if (f_phys_bdry_op)
      {
        f_phys_bdry_op->setPatchDataIndex(f_scratch_data_index);
        tbox::Pointer<hier::PatchLevel<spacedim>> level =
          hierarchy->getPatchLevel(level_number);
        for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
          {
            const tbox::Pointer<hier::Patch<spacedim>> patch =
              level->getPatch(p());
            tbox::Pointer<hier::PatchData<spacedim>> f_data =
              patch->getPatchData(f_scratch_data_index);
            f_phys_bdry_op->accumulateFromPhysicalBoundaryData(
              *patch, data_time, f_data->getGhostCellWidth());
          }
      }

    tbox::Pointer<hier::Variable<spacedim>> f_var;
    auto *var_db = hier::VariableDatabase<spacedim>::getDatabase();
    var_db->mapIndexToVariable(f_data_index, f_var);
    // Accumulate forces spread into patch ghost regions.
    {
      if (!ghost_data_accumulator)
        {
          // If we have multiple IBMethod objects we may end up with a wider
          // ghost region than the one required by this class. Hence, set the
          // ghost width by just picking whatever the data actually has at the
          // moment.
          const tbox::Pointer<hier::PatchLevel<spacedim>> level =
            hierarchy->getPatchLevel(level_number);
          const hier::IntVector<spacedim> gcw =
            level->getPatchDescriptor()
              ->getPatchDataFactory(f_scratch_data_index)
              ->getGhostCellWidth();

          ghost_data_accumulator.reset(new IBTK::SAMRAIGhostDataAccumulator(
            hierarchy, f_var, gcw, level_number, level_number));
        }
      ghost_data_accumulator->accumulateGhostData(f_scratch_data_index);
    }

    // Sum values back into the primary hierarchy.
    {
      auto f_primary_data_ops =
        extract_hierarchy_data_ops(f_var, primary_hierarchy);
      f_primary_data_ops->resetLevels(level_number, level_number);
      const auto f_primary_scratch_data_index =
        primary_eulerian_data_cache->getCachedPatchDataIndex(f_data_index);
      // we have to zero everything here since the scratch to primary
      // communication does not touch ghost cells, which may have junk
      fill_all(primary_hierarchy,
               f_primary_scratch_data_index,
               level_number,
               level_number,
               0.0);
      secondary_hierarchy
        .getSecondaryToPrimarySchedule(level_number,
                                       f_primary_scratch_data_index,
                                       f_scratch_data_index)
        .fillData(data_time);
      f_primary_data_ops->add(f_data_index,
                              f_data_index,
                              f_primary_scratch_data_index);
    }
    IBAMR_TIMER_STOP(t_spread_force);
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::applyGradientDetector(
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
    int                                               level_number,
    double /*error_data_time*/,
    int tag_index,
    bool /*initial_time*/,
    bool /*uses_richardson_extrapolation_too*/)
  {
    IBAMR_TIMER_START(t_apply_gradient_detector);
    // TODO: we should find a way to save the bboxes so they do not need to be
    // computed for each level that needs tagging - conceivably this could
    // happen in beginDataRedistribution() and the array can be cleared in
    // endDataRedistribution()
    for (const Part<dim, spacedim> &part : parts)
      {
        const DoFHandler<dim, spacedim> &dof_handler = part.get_dof_handler();
        MappingFEField<dim,
                       spacedim,
                       LinearAlgebra::distributed::Vector<double>>
                   mapping(dof_handler, part.get_position());
        const auto local_bboxes =
          compute_cell_bboxes<dim, spacedim, float>(dof_handler, mapping);
        // Like most other things this only works with p::S::T now
        const auto &tria =
          dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> &>(
            part.get_triangulation());
        const auto global_bboxes =
          collect_all_active_cell_bboxes(tria, local_bboxes);
        tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
          hierarchy->getPatchLevel(level_number);
        Assert(patch_level, ExcNotImplemented());
        tag_cells(global_bboxes, tag_index, patch_level);
      }
    IBAMR_TIMER_STOP(t_apply_gradient_detector);
  }

  //
  // Time stepping
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::preprocessIntegrateData(double current_time,
                                                     double new_time,
                                                     int /*num_cycles*/)
  {
    IBAMR_TIMER_START(t_preprocess_integrate_data);
    started_time_integration = true;
    part_vectors.begin_time_step(current_time, new_time);
    this->current_time = current_time;
    this->new_time     = new_time;
    this->half_time    = current_time + 0.5 * (new_time - current_time);
    IBAMR_TIMER_STOP(t_preprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::postprocessIntegrateData(double /*current_time*/,
                                                      double /*new_time*/,
                                                      int /*num_cycles*/)
  {
    IBAMR_TIMER_START(t_postprocess_integrate_data);
    this->current_time = std::numeric_limits<double>::quiet_NaN();
    this->new_time     = std::numeric_limits<double>::quiet_NaN();
    this->half_time    = std::numeric_limits<double>::quiet_NaN();

    // update positions and velocities:
    auto new_positions  = part_vectors.get_all_new_positions();
    auto new_velocities = part_vectors.get_all_new_velocities();
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        parts[part_n].set_position(std::move(new_positions[part_n]));
        parts[part_n].get_position().update_ghost_values();
        parts[part_n].set_velocity(std::move(new_velocities[part_n]));
        parts[part_n].get_velocity().update_ghost_values();
      }

    part_vectors.end_time_step();
    IBAMR_TIMER_STOP(t_postprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::forwardEulerStep(double current_time,
                                              double new_time)
  {
    const double dt = new_time - current_time;
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        // Set the position at the end time:
        LinearAlgebra::distributed::Vector<double> new_position(
          parts[part_n].get_partitioner());
        new_position = parts[part_n].get_position();
        new_position.add(dt, parts[part_n].get_velocity());
        part_vectors.set_position(part_n, new_time, std::move(new_position));

        // Set the position at the half time:
        LinearAlgebra::distributed::Vector<double> half_position(
          parts[part_n].get_partitioner());
        half_position.add(0.5,
                          part_vectors.get_position(part_n, current_time),
                          0.5,
                          part_vectors.get_position(part_n, new_time));
        part_vectors.set_position(part_n, half_time, std::move(half_position));
      }
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::backwardEulerStep(double current_time,
                                               double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::midpointStep(double current_time, double new_time)
  {
    const double dt = new_time - current_time;
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        // Set the position at the end time:
        LinearAlgebra::distributed::Vector<double> new_position(
          parts[part_n].get_partitioner());
        new_position = parts[part_n].get_position();
        new_position.add(dt, part_vectors.get_velocity(part_n, half_time));
        part_vectors.set_position(part_n, new_time, std::move(new_position));

        // Set the position at the half time:
        LinearAlgebra::distributed::Vector<double> half_position(
          parts[part_n].get_partitioner());
        half_position.add(0.5,
                          part_vectors.get_position(part_n, current_time),
                          0.5,
                          part_vectors.get_position(part_n, new_time));
        part_vectors.set_position(part_n, half_time, std::move(half_position));
      }
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::trapezoidalStep(double current_time,
                                             double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }

  //
  // Mechanics
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::computeLagrangianForce(double data_time)
  {
    IBAMR_TIMER_START(t_compute_lagrangian_force);
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const Part<dim, spacedim> &                part = parts[part_n];
        LinearAlgebra::distributed::Vector<double> force(
          part.get_partitioner()),
          force_rhs(part.get_partitioner());

        // The velocity isn't available at data_time so use current_time -
        // IBFEMethod does this too.
        const auto &position = part_vectors.get_position(part_n, data_time);
        const auto &velocity = part_vectors.get_velocity(part_n, current_time);
        // Unlike velocity interpolation and force spreading we actually need
        // the ghost values in the native partitioning, so make sure they are
        // available
        position.update_ghost_values();
        velocity.update_ghost_values();
        compute_volumetric_pk1_load_vector(part.get_dof_handler(),
                                           part.get_mapping(),
                                           part.get_stress_contributions(),
                                           position,
                                           velocity,
                                           force_rhs);
        force_rhs.compress(VectorOperation::add);

        SolverControl control(1000, 1e-14 * force_rhs.l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
        // TODO - implement initial guess stuff here
        cg.solve(part.get_mass_operator(),
                 force,
                 force_rhs,
                 part.get_mass_preconditioner());
        part_vectors.set_force(part_n, data_time, std::move(force));
      }
    IBAMR_TIMER_STOP(t_compute_lagrangian_force);
  }

  //
  // Data redistribution
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::reinit_interactions()
  {
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const Part<dim, spacedim> &part = parts[part_n];

        const auto &tria =
          dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> &>(
            part.get_triangulation());
        const DoFHandler<dim, spacedim> &dof_handler = part.get_dof_handler();
        MappingFEField<dim,
                       spacedim,
                       LinearAlgebra::distributed::Vector<double>>
          mapping(dof_handler, part.get_position());
        IBAMR_TIMER_START(t_reinit_interactions_bboxes);
        const auto local_bboxes =
          compute_cell_bboxes<dim, spacedim, float>(dof_handler, mapping);
        const auto global_bboxes =
          collect_all_active_cell_bboxes(tria, local_bboxes);
        IBAMR_TIMER_STOP(t_reinit_interactions_bboxes);

        IBAMR_TIMER_START(t_reinit_interactions_edges);
        const auto local_edge_lengths = compute_longest_edge_lengths(
          tria, mapping, QGauss<1>(dof_handler.get_fe().tensor_degree()));
        const auto global_edge_lengths =
          collect_longest_edge_lengths(tria, local_edge_lengths);
        IBAMR_TIMER_STOP(t_reinit_interactions_edges);

        IBAMR_TIMER_START(t_reinit_interactions_objects);
        interactions[part_n]->reinit(
          tria,
          global_bboxes,
          global_edge_lengths,
          secondary_hierarchy.getSecondaryHierarchy(),
          primary_hierarchy->getFinestLevelNumber());
        // TODO - we should probably add a reinit() function that sets up the
        // DoFHandler we always need
        interactions[part_n]->add_dof_handler(part.get_dof_handler());
        IBAMR_TIMER_STOP(t_reinit_interactions_objects);
      }
  }


  template <int dim, int spacedim>
  void IFEDMethod<dim, spacedim>::beginDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_begin_data_redistribution);
    // This function is called before initializePatchHierarchy is - in that case
    // we don't have a hierarchy, so we don't have any data, and there is naught
    // to do
    if (primary_hierarchy)
      {
        // Weird things happen when we coarsen and refine if some levels are not
        // present, so fill them all in with zeros to start
        const int max_ln = primary_hierarchy->getFinestLevelNumber();
        for (int ln = 0; ln <= max_ln; ++ln)
          {
            tbox::Pointer<hier::PatchLevel<spacedim>> primary_level =
              primary_hierarchy->getPatchLevel(ln);
            tbox::Pointer<hier::PatchLevel<spacedim>> secondary_level =
              secondary_hierarchy.getSecondaryHierarchy()->getPatchLevel(ln);
            if (!primary_level->checkAllocated(
                  lagrangian_workload_current_index))
              primary_level->allocatePatchData(
                lagrangian_workload_current_index);
            if (!secondary_level->checkAllocated(
                  lagrangian_workload_current_index))
              secondary_level->allocatePatchData(
                lagrangian_workload_current_index);
          }

        fill_all(secondary_hierarchy.getSecondaryHierarchy(),
                 lagrangian_workload_current_index,
                 0,
                 max_ln);

        // start:
        std::vector<std::unique_ptr<TransactionBase>> transactions;
        for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
          {
            const Part<dim, spacedim> &part = parts[part_n];
            transactions.emplace_back(interactions[part_n]->add_workload_start(
              lagrangian_workload_current_index,
              part.get_position(),
              part.get_dof_handler()));
          }

        // Compute:
        for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
          transactions[part_n] =
            interactions[part_n]->add_workload_intermediate(
              std::move(transactions[part_n]));

        // Finish:
        for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
          interactions[part_n]->add_workload_finish(
            std::move(transactions[part_n]));

        // Move to primary hierarchy (we will read it back in
        // endDataRedistribution)
        fill_all(primary_hierarchy,
                 lagrangian_workload_current_index,
                 0,
                 max_ln);

        secondary_hierarchy
          .getSecondaryToPrimarySchedule(max_ln,
                                         lagrangian_workload_current_index,
                                         lagrangian_workload_current_index)
          .fillData(0.0);
      }

    // Clear a few things that depend on the current hierarchy:
    ghost_data_accumulator.reset();
    IBAMR_TIMER_STOP(t_begin_data_redistribution);
  }

  template <int dim, int spacedim>
  void IFEDMethod<dim, spacedim>::endDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_end_data_redistribution);
    // same as beginDataRedistribution
    if (primary_hierarchy)
      {
        secondary_hierarchy.reinit(primary_hierarchy->getFinestLevelNumber(),
                                   primary_hierarchy->getFinestLevelNumber(),
                                   primary_hierarchy,
                                   lagrangian_workload_current_index);

        reinit_interactions();

        if (input_db->getBoolWithDefault("enable_logging", true) &&
            (started_time_integration ||
             (!started_time_integration &&
              !input_db->getBoolWithDefault("skip_initial_workload", false))))
          {
            const int ln            = primary_hierarchy->getFinestLevelNumber();
            auto      secondary_ops = extract_hierarchy_data_ops(
              lagrangian_workload_var,
              secondary_hierarchy.getSecondaryHierarchy());
            secondary_ops->resetLevels(ln, ln);
            const double work =
              secondary_ops->L1Norm(lagrangian_workload_current_index,
                                    IBTK::invalid_index,
                                    true);
            const std::vector<double> all_work =
              Utilities::MPI::all_gather(IBTK::IBTK_MPI::getCommunicator(),
                                         work);

            const int  n_processes   = IBTK::IBTK_MPI::getNodes();
            const auto right_padding = std::size_t(std::log10(n_processes)) + 1;
            if (IBTK::IBTK_MPI::getRank() == 0)
              {
                for (int rank = 0; rank < n_processes; ++rank)
                  {
                    tbox::plog << "IFEDMethod::endDataRedistribution(): "
                               << "workload estimate on processor "
                               << std::setw(right_padding) << std::left << rank
                               << " = " << all_work[rank] << '\n';
                  }
                tbox::plog << "IFEDMethod::endDataRedistribution(): "
                           << "total workload = "
                           << std::accumulate(all_work.begin(),
                                              all_work.end(),
                                              0.0)
                           << std::endl;
              }
          }

        // IBTK::HierarchyIntegrator (which controls these data indices) will
        // exchange pointers between the new and current states during
        // timestepping. In particular: it will swap new and current, deallocate
        // new, reallocate new, and repeat. This causes (since we only touch
        // this variable in regrids) this data to get cleared, so save a copy
        // for plotting purposes.

        // TODO - implement a utility function for copying
        auto primary_ops = extract_hierarchy_data_ops(lagrangian_workload_var,
                                                      primary_hierarchy);
        const int max_ln = primary_hierarchy->getFinestLevelNumber();
        primary_ops->resetLevels(0, max_ln);
        for (int ln = 0; ln <= max_ln; ++ln)
          {
            tbox::Pointer<hier::PatchLevel<spacedim>> primary_level =
              primary_hierarchy->getPatchLevel(ln);
            if (!primary_level->checkAllocated(lagrangian_workload_plot_index))
              primary_level->allocatePatchData(lagrangian_workload_plot_index);
          }
        primary_ops->copyData(lagrangian_workload_plot_index,
                              lagrangian_workload_current_index,
                              false);
      }
    IBAMR_TIMER_STOP(t_end_data_redistribution);
  }

  //
  // Book-keeping
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::putToDatabase(tbox::Pointer<tbox::Database> db)
  {
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        std::ostringstream              out_str;
        boost::archive::binary_oarchive oarchive(out_str);
        parts[part_n].save(oarchive, 0);
        // TODO - with C++20 we can use view() instead of str() and skip this
        // copy
        const std::string out = out_str.str();
        save_binary("part_" + std::to_string(part_n),
                    out.c_str(),
                    out.c_str() + out.size(),
                    db);
      }
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::registerEulerianVariables()
  {
    // we need ghosts for CONSERVATIVE_LINEAR_REFINE
    const hier::IntVector<spacedim> ghosts = 1;
    lagrangian_workload_var = new pdat::CellVariable<spacedim, double>(
      object_name + "::lagrangian_workload");
    registerVariable(lagrangian_workload_current_index,
                     lagrangian_workload_new_index,
                     lagrangian_workload_scratch_index,
                     lagrangian_workload_var,
                     ghosts,
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE");

    auto *var_db  = hier::VariableDatabase<spacedim>::getDatabase();
    auto  context = var_db->getContext(object_name);
    lagrangian_workload_plot_index =
      var_db->registerVariableAndContext(lagrangian_workload_var,
                                         context,
                                         ghosts);
  }



  template <int dim, int spacedim>
  const hier::IntVector<spacedim> &
  IFEDMethod<dim, spacedim>::getMinimumGhostCellWidth() const
  {
    static hier::IntVector<spacedim> gcw;
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const int ghost_width =
          IBTK::LEInteractor::getMinimumGhostWidth(ib_kernels[part_n]);
        for (int i = 0; i < spacedim; ++i)
          gcw[i] = std::max(gcw[i], ghost_width);
      }
    return gcw;
  }

  template class IFEDMethod<NDIM - 1, NDIM>;
  template class IFEDMethod<NDIM, NDIM>;
} // namespace fdl
