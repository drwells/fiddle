#include <fiddle/base/samrai_utilities.h>
#include <fiddle/base/utilities.h>

#include <fiddle/grid/box_utilities.h>

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
    tbox::Pointer<tbox::Database>      input_db,
    std::vector<Part<dim, spacedim>> &&input_parts,
    const bool                         register_for_restart)
    : object_name(object_name)
    , register_for_restart(register_for_restart)
    , input_db(copy_database(input_db))
    , started_time_integration(false)
    , current_time(std::numeric_limits<double>::signaling_NaN())
    , half_time(std::numeric_limits<double>::signaling_NaN())
    , new_time(std::numeric_limits<double>::signaling_NaN())
    , parts(std::move(input_parts))
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

    // Ignore RestartManager unless requested
    if (register_for_restart)
      {
        if (tbox::RestartManager::getManager()->isFromRestart())
          {
            auto restart_db =
              tbox::RestartManager::getManager()->getRootDatabase();
            if (restart_db->isDatabase(object_name))
              {
                auto db = restart_db->getDatabase(object_name);
                for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
                  {
                    const std::string key = "part_" + std::to_string(part_n);
                    Assert(db->keyExists(key),
                           ExcMessage("Couldn't find key " + key +
                                      " in the restart database"));
                    // Same note from putToDatabase applies here
                    const std::string base64 = db->getString(key);
                    std::string serialization =
                      decode_base64(base64.c_str(),
                                    base64.c_str() + base64.size());
                    std::istringstream in_str(serialization);
                    boost::archive::binary_iarchive iarchive(in_str);
                    parts[part_n].load(iarchive, 0);
                  }
              }
            else
              {
                Assert(false,
                       ExcMessage("The restart database does not contain key " +
                                  object_name));
              }
          }
        else
          {
            tbox::RestartManager::getManager()->registerRestartItem(object_name,
                                                                    this);
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
            u_data_index,
            part.get_dof_handler(),
            get_position(part_n, data_time),
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
        if (std::abs(data_time - half_time) < 1e-14)
          {
            half_velocity_vectors.resize(parts.size());
            half_velocity_vectors[part_n] = std::move(velocity);
          }
        else if (std::abs(data_time - new_time) < 1e-14)
          {
            new_velocity_vectors.resize(parts.size());
            new_velocity_vectors[part_n] = std::move(velocity);
          }
        else
          Assert(false, ExcFDLNotImplemented());
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
          f_scratch_data_index,
          get_position(part_n, data_time),
          part.get_dof_handler(),
          part.get_mapping(),
          part.get_dof_handler(),
          get_force(part_n, data_time)));
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
    hier::VariableDatabase<spacedim>::getDatabase()->mapIndexToVariable(
      f_data_index, f_var);
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
    this->current_time       = current_time;
    this->new_time           = new_time;
    this->half_time          = current_time + 0.5 * (new_time - current_time);
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
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        parts[part_n].set_position(std::move(new_position_vectors[part_n]));
        parts[part_n].get_position().update_ghost_values();
        parts[part_n].set_velocity(std::move(new_velocity_vectors[part_n]));
        parts[part_n].get_velocity().update_ghost_values();
      }

    half_position_vectors.clear();
    new_position_vectors.clear();
    half_velocity_vectors.clear();
    new_velocity_vectors.clear();

    current_force_vectors.clear();
    half_force_vectors.clear();
    new_force_vectors.clear();
    IBAMR_TIMER_STOP(t_postprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::forwardEulerStep(double current_time,
                                              double new_time)
  {
    const double dt = new_time - current_time;
    half_position_vectors.resize(parts.size());
    new_position_vectors.resize(parts.size());
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        // Set the position at the end time:
        new_position_vectors[part_n] = parts[part_n].get_position();
        new_position_vectors[part_n].add(dt, parts[part_n].get_velocity());
        new_position_vectors[part_n].update_ghost_values();

        // Set the position at the half time:
        half_position_vectors[part_n].reinit(parts[part_n].get_position(),
                                             /*omit_zeroing_entries = */ true);
        half_position_vectors[part_n].equ(0.5, parts[part_n].get_position());
        half_position_vectors[part_n].add(0.5, new_position_vectors[part_n]);
        half_position_vectors[part_n].update_ghost_values();
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
    half_position_vectors.resize(parts.size());
    new_position_vectors.resize(parts.size());
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        // Set the position at the end time:
        new_position_vectors[part_n] = parts[part_n].get_position();
        Assert(part_n < half_velocity_vectors.size(), ExcFDLInternalError());
        new_position_vectors[part_n].add(dt, half_velocity_vectors[part_n]);

        // Set the position at the half time:
        half_position_vectors[part_n].reinit(parts[part_n].get_position(),
                                             /*omit_zeroing_entries = */ true);
        half_position_vectors[part_n].equ(0.5, parts[part_n].get_position());
        half_position_vectors[part_n].add(0.5, new_position_vectors[part_n]);
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
    std::vector<LinearAlgebra::distributed::Vector<double>> *force_vectors =
      nullptr;
    if (std::abs(data_time - current_time) < 1e-14)
      force_vectors = &current_force_vectors;
    else if (std::abs(data_time - half_time) < 1e-14)
      force_vectors = &half_force_vectors;
    else if (std::abs(data_time - new_time) < 1e-14)
      force_vectors = &new_force_vectors;

    Assert(force_vectors != nullptr, ExcFDLInternalError());
    if (force_vectors)
      {
        force_vectors->resize(0);
        for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
          {
            const Part<dim, spacedim> &part = parts[part_n];
            force_vectors->emplace_back(part.get_partitioner());
            auto &force_vector = force_vectors->back();
            LinearAlgebra::distributed::Vector<double> force_rhs = force_vector;

            // TODO - ask Boyce about the velocity. It's not available at
            // data_time at the point where this function is called. This isn't
            // critical - the velocity is mostly used to implement penalties so
            // being inaccurate shouldn't sink us
            compute_volumetric_pk1_load_vector(part.get_dof_handler(),
                                               part.get_mapping(),
                                               part.get_stress_contributions(),
                                               get_position(part_n, data_time),
                                               get_velocity(part_n,
                                                            current_time),
                                               force_rhs);
            force_rhs.compress(VectorOperation::add);

            SolverControl control(1000, 1e-14 * force_rhs.l2_norm());
            SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
            // TODO - implement initial guess stuff here
            cg.solve(part.get_mass_operator(),
                     force_vector,
                     force_rhs,
                     part.get_mass_preconditioner());
          }
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
        const auto local_bboxes =
          compute_cell_bboxes<dim, spacedim, float>(dof_handler, mapping);

        const auto global_bboxes =
          collect_all_active_cell_bboxes(tria, local_bboxes);

        interactions[part_n]->reinit(
          tria,
          global_bboxes,
          secondary_hierarchy.getSecondaryHierarchy(),
          primary_hierarchy->getFinestLevelNumber());
        // TODO - we should probably add a reinit() function that sets up the
        // DoFHandler we always need
        interactions[part_n]->add_dof_handler(part.get_dof_handler());
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
      }
    IBAMR_TIMER_STOP(t_end_data_redistribution);
  }

  //
  // Book-keeping
  //


  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  IFEDMethod<dim, spacedim>::get_position(const unsigned int part_n,
                                          const double       time) const
  {
    if (std::abs(time - current_time) < 1e-12)
      return parts[part_n].get_position();
    if (std::abs(time - half_time) < 1e-12)
      {
        Assert(part_n < half_position_vectors.size(),
               ExcMessage(
                 "The requested position vector has not been calculated."));
        return half_position_vectors[part_n];
      }
    if (std::abs(time - new_time) < 1e-12)
      {
        Assert(part_n < new_position_vectors.size(),
               ExcMessage(
                 "The requested position vector has not been calculated."));
        return new_position_vectors[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n].get_position();
  }

  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  IFEDMethod<dim, spacedim>::get_velocity(const unsigned int part_n,
                                          const double       time) const
  {
    if (std::abs(time - current_time) < 1e-12)
      return parts[part_n].get_velocity();
    if (std::abs(time - half_time) < 1e-12)
      {
        Assert(part_n < half_velocity_vectors.size(),
               ExcMessage(
                 "The requested velocity vector has not been calculated."));
        return half_velocity_vectors[part_n];
      }
    if (std::abs(time - new_time) < 1e-12)
      {
        Assert(part_n < new_velocity_vectors.size(),
               ExcMessage(
                 "The requested velocity vector has not been calculated."));
        return new_velocity_vectors[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n].get_position();
  }

  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  IFEDMethod<dim, spacedim>::get_force(const unsigned int part_n,
                                       const double       time) const
  {
    if (std::abs(time - current_time) < 1e-12)
      {
        Assert(part_n < current_force_vectors.size(),
               ExcMessage(
                 "The requested force vector has not been calculated."));
        return current_force_vectors[part_n];
      }
    if (std::abs(time - half_time) < 1e-12)
      {
        Assert(part_n < half_force_vectors.size(),
               ExcMessage(
                 "The requested force vector has not been calculated."));
        return half_force_vectors[part_n];
      }
    if (std::abs(time - new_time) < 1e-12)
      {
        Assert(part_n < new_force_vectors.size(),
               ExcMessage(
                 "The requested force vector has not been calculated."));
        return new_force_vectors[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n].get_position();
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::putToDatabase(tbox::Pointer<tbox::Database> db)
  {
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        std::ostringstream              out_str;
        boost::archive::binary_oarchive oarchive(out_str);
        parts[part_n].save(oarchive, 0);
        // Unfortunately, SAMRAI doesn't understand that a std::string can
        // contain NUL characters (it uses c_str(), which is wrong) so we have
        // to do an extra translation to get around its bugs:
        //
        // TODO - with C++20 we can use view() instead of str() and skip one
        // more copy
        const std::string          out = out_str.str();
        db->putString("part_" + std::to_string(part_n),
                      encode_base64(out.c_str(), out.c_str() + out.size()));
      }
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::registerEulerianVariables()
  {
    // we need ghosts for CONSERVATIVE_LINEAR_REFINE
    const hier::IntVector<spacedim> ghosts = 1;
    lagrangian_workload_var =
      new pdat::CellVariable<spacedim, double>("::lagrangian_workload");
    registerVariable(lagrangian_workload_current_index,
                     lagrangian_workload_new_index,
                     lagrangian_workload_scratch_index,
                     lagrangian_workload_var,
                     ghosts,
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE");
  }



  template <int dim, int spacedim>
  const hier::IntVector<spacedim> &
  IFEDMethod<dim, spacedim>::getMinimumGhostCellWidth() const
  {
    // Like elsewhere, we are hard-coding in bspline 3 for now
    const std::string kernel_name = "BSPLINE_3";
    const int         ghost_width =
      IBTK::LEInteractor::getMinimumGhostWidth(kernel_name);
    static hier::IntVector<spacedim> gcw;
    for (int i = 0; i < spacedim; ++i)
      gcw[i] = ghost_width;
    return gcw;
  }

  template class IFEDMethod<NDIM - 1, NDIM>;
  template class IFEDMethod<NDIM, NDIM>;
} // namespace fdl
