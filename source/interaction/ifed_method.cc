#include <fiddle/base/samrai_utilities.h>
#include <fiddle/base/utilities.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/grid_utilities.h>

#include <fiddle/interaction/elemental_interaction.h>
#include <fiddle/interaction/ifed_method.h>
#include <fiddle/interaction/interaction_utilities.h>
#include <fiddle/interaction/nodal_interaction.h>

#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/lac/solver_cg.h>

#include <ibamr/IBHierarchyIntegrator.h>
#include <ibamr/ibamr_utilities.h>

#include <ibtk/IBTK_MPI.h>
#include <ibtk/ibtk_utilities.h>

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
  static tbox::Timer *t_interpolate_velocity_rhs;
  static tbox::Timer *t_interpolate_velocity_solve;
  static tbox::Timer *t_compute_lagrangian_force;
  static tbox::Timer *t_compute_lagrangian_force_pk1;
  static tbox::Timer *t_compute_lagrangian_force_solve;
  static tbox::Timer *t_spread_force;
  static tbox::Timer *t_max_point_displacement;
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
    const std::string                 &object_name,
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
    , ghosts(0)
    , secondary_hierarchy(object_name + "::secondary_hierarchy",
                          input_db->getDatabase("GriddingAlgorithm"),
                          input_db->getDatabase("LoadBalancer"))
  {
    // IBAMR does not support using threads so unconditionally disable them
    // here.
    MultithreadInfo::set_thread_limit(1);

    auto init_regrid_positions = [](auto &vectors, const auto &collection)
    {
      for (const auto &part : collection)
        vectors.push_back(part.get_position());
    };
    init_regrid_positions(positions_at_last_regrid, parts);
    init_regrid_positions(penalty_positions_at_last_regrid, penalty_parts);

    const std::string interaction =
      input_db->getStringWithDefault("interaction", "ELEMENTAL");
    if (interaction == "ELEMENTAL")
      {
        // IBFEMethod uses this value - lower values aren't guaranteed to work.
        // If dx = dX then we can use a lower density.
        const double density =
          input_db->getDoubleWithDefault("IB_point_density", 2.0);

        // Default to minimum density:
        auto        density_kind = DensityKind::Minimum;
        std::string density_kind_string =
          input_db->getStringWithDefault("density_kind", "Minimum");
        std::transform(density_kind_string.begin(),
                       density_kind_string.end(),
                       density_kind_string.begin(),
                       [](const unsigned char c) { return std::tolower(c); });
        if (density_kind_string == "minimum")
          density_kind = DensityKind::Minimum;
        else if (density_kind_string == "average")
          density_kind = DensityKind::Average;
        else
          AssertThrow(false, ExcFDLNotImplemented());

        auto init_elemental = [&](auto       &inters,
                                  auto       &guess_1,
                                  auto       &guess_2,
                                  const auto &collection)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          for (unsigned int i = 0; i < collection.size(); ++i)
            {
              const unsigned int n_points_1D =
                collection[i].get_dof_handler().get_fe().tensor_degree() + 1;
              inters.emplace_back(
                std::make_unique<ElementalInteraction<structdim, spacedim>>(
                  n_points_1D, density, density_kind));
              guess_1.emplace_back(
                input_db->getIntegerWithDefault("n_guess_vectors", 3));
              guess_2.emplace_back(
                input_db->getIntegerWithDefault("n_guess_vectors", 3));
            }
        };
        init_elemental(interactions, force_guesses, velocity_guesses, parts);
        init_elemental(penalty_interactions, penalty_force_guesses, penalty_velocity_guesses, penalty_parts);
      }
    else if (interaction == "NODAL")
      {
        auto init_nodal = [&](auto &inters, const auto &collection)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          for (unsigned int i = 0; i < collection.size(); ++i)
            inters.emplace_back(
              std::make_unique<NodalInteraction<structdim, spacedim>>());
        };
        init_nodal(interactions, parts);
        init_nodal(penalty_interactions, penalty_parts);
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("unsupported interaction type " + interaction +
                               "."));
      }

    auto do_kernel = [&](const std::string &key,
                         const auto &collection,
                         std::vector<std::string> &kernels)
    {
      if (collection.size() > 0)
        {
          AssertThrow(input_db->keyExists(key),
            ExcMessage(key + " must be set in the input database."));
          // values in SAMRAI databases are always arrays, possibly of
          // length 1
          const int n_ib_kernels = input_db->getArraySize(key);
          AssertThrow(n_ib_kernels == 1 ||
                      n_ib_kernels == static_cast<int>(collection.size()),
                      ExcMessage("The number of specified IB kernels should "
                                 "either be 1 or equal the number of (penalty) "
                                 "parts."));
          kernels.resize(n_ib_kernels);
          input_db->getStringArray(key,
                                   kernels.data(),
                                   static_cast<int>(kernels.size()));
          // ib_kernels is either length 1 or length collection.size(): deal
          // with the first case.
          if (n_ib_kernels == 1)
            {
              kernels.resize(collection.size());
              std::fill(kernels.begin() + 1, kernels.end(), kernels.front());
            }

          // now that we know that, we know the ghost requirements
          for (const std::string &kernel : kernels)
            {
              const int ghost_width =
                IBTK::LEInteractor::getMinimumGhostWidth(kernel);
              for (int d = 0; d < spacedim; ++d)
                ghosts[d] = std::max(ghosts[d], ghost_width);
            }
        }
    };
    do_kernel("IB_kernel", parts, ib_kernels);
    do_kernel("penalty_IB_kernel", penalty_parts, penalty_ib_kernels);

    auto set_timer = [&](const char *name)
    { return tbox::TimerManager::getManager()->getTimer(name); };

    t_preprocess_integrate_data =
      set_timer("fdl::IFEDMethod::preprocessIntegrateData()");
    t_postprocess_integrate_data =
      set_timer("fdl::IFEDMethod::postprocessIntegrateData()");
    t_interpolate_velocity =
      set_timer("fdl::IFEDMethod::interpolateVelocity()");
    t_interpolate_velocity_rhs =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[rhs]");
    t_interpolate_velocity_solve =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[solve]");
    t_compute_lagrangian_force =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()");
    t_compute_lagrangian_force_pk1 =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[pk1]");
    t_compute_lagrangian_force_solve =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[solve]");
    t_spread_force = set_timer("fdl::IFEDMethod::spreadForce()");
    t_max_point_displacement =
      set_timer("fdl::IFEDMethod::getMaxPointDisplacement()");
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
                auto db      = restart_db->getDatabase(object_name);
                auto do_load = [&](auto &collection, const std::string &prefix)
                {
                  for (unsigned int i = 0; i < collection.size(); ++i)
                    {
                      const std::string key = prefix + std::to_string(i);
                      AssertThrow(db->keyExists(key),
                                  ExcMessage("Couldn't find key " + key +
                                             " in the restart database"));
                      const std::string  serialization = load_binary(key, db);
                      std::istringstream in_str(serialization);
                      boost::archive::binary_iarchive iarchive(in_str);
                      collection[i].load(iarchive, 0);
                    }
                };
                do_load(parts, "part_");
                do_load(penalty_parts, "penalty_part_");
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
          &u_ghost_fill_scheds,
    double data_time)
  {
    IBAMR_TIMER_START(t_interpolate_velocity);
    (void)u_synch_scheds;
    (void)u_ghost_fill_scheds;

    // Update the secondary hierarchy:
    secondary_hierarchy.transferPrimaryToSecondary(
      primary_hierarchy->getFinestLevelNumber(),
      u_data_index,
      u_data_index,
      data_time,
      d_ib_solver->getVelocityPhysBdryOp());

    // start:
    IBAMR_TIMER_START(t_interpolate_velocity_rhs);
    auto setup_transaction = [&](const auto &collection,
                                 const auto &interactions,
                                 const auto &kernels,
                                 const auto &vectors,
                                 auto       &transactions,
                                 auto       &rhs_vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          const auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          rhs_vectors.emplace_back(part.get_partitioner());
          transactions.emplace_back(
            interactions[i]->compute_projection_rhs_start(
              kernels[i],
              u_data_index,
              part.get_dof_handler(),
              vectors.get_position(i, data_time),
              part.get_dof_handler(),
              part.get_mapping(),
              rhs_vectors[i]));
        }
    };
    // we emplace_back so use a deque to keep pointers valid
    std::vector<std::unique_ptr<TransactionBase>>          transactions, penalty_transactions;
    std::deque<LinearAlgebra::distributed::Vector<double>> rhs_vecs, penalty_rhs_vecs;
    setup_transaction(
      parts, interactions, ib_kernels, part_vectors, transactions, rhs_vecs);
    setup_transaction(
      penalty_parts, penalty_interactions, penalty_ib_kernels, penalty_part_vectors, penalty_transactions, penalty_rhs_vecs);

    // Compute:
    auto compute_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        transactions[i] = interactions[i]->compute_projection_rhs_intermediate(
          std::move(transactions[i]));
    };
    compute_transaction(interactions, transactions);
    compute_transaction(penalty_interactions, penalty_transactions);

    // Collect:
    auto collect_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        interactions[i]->compute_projection_rhs_finish(
          std::move(transactions[i]));
    };
    collect_transaction(interactions, transactions);
    collect_transaction(penalty_interactions, penalty_transactions);

    // We cannot start the linear solve without first finishing this, so use a
    // barrier to keep the timers accurate
    const int ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
    AssertThrowMPI(ierr);
    IBAMR_TIMER_STOP(t_interpolate_velocity_rhs);

    // Project:
    IBAMR_TIMER_START(t_interpolate_velocity_solve);
    auto do_solve = [&](const auto &collection,
                        auto       &vectors,
                        auto       &guesses,
                        auto       &rhs_vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          if (interactions[i]->projection_is_interpolation())
            {
              // If projection is actually interpolation we have a lot less to
              // do
              vectors.set_velocity(i, data_time, std::move(rhs_vectors[i]));
            }
          else
            {
              const auto   &part = collection[i];
              SolverControl control(
                input_db->getIntegerWithDefault("solver_iterations", 100),
                input_db->getDoubleWithDefault("solver_relative_tolerance",
                                               1e-6) *
                  rhs_vectors[i].l2_norm());
              SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
              LinearAlgebra::distributed::Vector<double>           velocity(
                part.get_partitioner());

              // If we mess up the matrix-free implementation will fix our
              // partitioner: make sure we catch that case here
              Assert(velocity.get_partitioner() == part.get_partitioner(),
                     ExcFDLInternalError());
              guesses[i].guess(velocity, rhs_vectors[i]);
              cg.solve(part.get_mass_operator(),
                       velocity,
                       rhs_vectors[i],
                       part.get_mass_preconditioner());
              guesses[i].submit(velocity, rhs_vectors[i]);
              // Same
              Assert(velocity.get_partitioner() == part.get_partitioner(),
                     ExcFDLInternalError());
              vectors.set_velocity(i, data_time, std::move(velocity));

              if (input_db->getBoolWithDefault("log_solver_iterations", false))
                {
                  tbox::plog << "IFEDMethod::interpolateVelocity(): "
                             << "SolverCG<> converged in "
                             << control.last_step() << " steps." << std::endl;
                }
            }
        }
    };
    do_solve(parts, part_vectors, velocity_guesses, rhs_vecs);
    do_solve(penalty_parts, penalty_part_vectors, penalty_velocity_guesses, penalty_rhs_vecs);
    IBAMR_TIMER_STOP(t_interpolate_velocity_solve);
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
    auto setup_transaction = [&](const auto &collection,
                                 const auto &interactions,
                                 const auto &kernels,
                                 const auto &vectors,
                                 auto       &transactions)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          const auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          transactions.emplace_back(interactions[i]->compute_spread_start(
            kernels[i],
            f_scratch_data_index,
            vectors.get_position(i, data_time),
            part.get_dof_handler(),
            part.get_mapping(),
            part.get_dof_handler(),
            vectors.get_force(i, data_time)));
        }
    };
    std::vector<std::unique_ptr<TransactionBase>> transactions, penalty_transactions;
    setup_transaction(parts, interactions, part_vectors, transactions);
    setup_transaction(penalty_parts, penalty_interactions, penalty_part_vectors, penalty_transactions);

    // Compute:
    auto compute_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        transactions[i] = interactions[i]->compute_spread_intermediate(
          std::move(transactions[i]));
    };
    compute_transaction(interactions, transactions);
    compute_transaction(penalty_interactions, penalty_transactions);

    // Collect:
    auto collect_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        interactions[i]->compute_spread_finish(std::move(transactions[i]));
    };
    collect_transaction(interactions, transactions);
    collect_transaction(penalty_interactions, penalty_transactions);

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
      secondary_hierarchy.transferSecondaryToPrimary(
        level_number,
        f_primary_scratch_data_index,
        f_scratch_data_index,
        data_time);
      f_primary_data_ops->add(f_data_index,
                              f_data_index,
                              f_primary_scratch_data_index);
    }
    IBAMR_TIMER_STOP(t_spread_force);
  }


  template <int dim, int spacedim>
  double
  IFEDMethod<dim, spacedim>::getMaxPointDisplacement() const
  {
    IBAMR_TIMER_START(t_max_point_displacement);
    double max_displacement = 0;

    auto max_op = [&](const auto &collection, const auto &regrid_positions)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          AssertDimension(collection.size(), regrid_positions.size());
          const auto &ref_position = regrid_positions[i];
          const auto &position     = collection[i].get_position();
          const auto  local_size   = position.locally_owned_size();
          for (unsigned int j = 0; j < local_size; ++j)
            max_displacement = std::max(max_displacement,
                                        std::abs(ref_position.local_element(j) -
                                                 position.local_element(j)));
        }
    };
    max_op(parts, positions_at_last_regrid);
    max_op(penalty_parts, penalty_positions_at_last_regrid);
    max_displacement =
      Utilities::MPI::max(max_displacement, IBTK::IBTK_MPI::getCommunicator());

    return max_displacement /
           IBTK::get_min_patch_dx(
             dynamic_cast<const hier::PatchLevel<spacedim> &>(
               *primary_hierarchy->getPatchLevel(
                 primary_hierarchy->getFinestLevelNumber())));
    IBAMR_TIMER_STOP(t_max_point_displacement);
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
    auto do_tag = [&](const auto &collection)
    {
      for (const auto &part : collection)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          MappingFEField<structdim,
                         spacedim,
                         LinearAlgebra::distributed::Vector<double>>
                     mapping(part.get_dof_handler(), part.get_position());
          const auto local_bboxes =
            compute_cell_bboxes<structdim, spacedim, float>(
              part.get_dof_handler(), mapping);
          // Like most other things this only works with p::S::T now
          const auto &tria = dynamic_cast<
            const parallel::shared::Triangulation<structdim, spacedim> &>(
            part.get_triangulation());
          const auto global_bboxes =
            collect_all_active_cell_bboxes(tria, local_bboxes);
          tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
            hierarchy->getPatchLevel(level_number);
          Assert(patch_level, ExcNotImplemented());
          tag_cells(global_bboxes, tag_index, patch_level);
        }
    };

    do_tag(parts);
    do_tag(penalty_parts);
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
    penalty_part_vectors.begin_time_step(current_time, new_time);
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
    auto do_set = [](auto &collection, auto &positions, auto &velocities)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          part.set_position(std::move(positions[i]));
          part.get_position().update_ghost_values();
          part.set_velocity(std::move(velocities[i]));
          part.get_velocity().update_ghost_values();
        }
    };
    auto new_positions  = part_vectors.get_all_new_positions();
    auto new_velocities = part_vectors.get_all_new_velocities();
    auto penalty_new_positions  = part_vectors.get_all_new_positions();
    auto penalty_new_velocities = part_vectors.get_all_new_velocities();
    do_set(parts, new_positions, new_velocities);
    do_set(penalty_parts, penalty_new_positions, penalty_new_velocities);

    part_vectors.end_time_step();
    penalty_part_vectors.end_time_step();
    IBAMR_TIMER_STOP(t_postprocess_integrate_data);
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::forwardEulerStep(double current_time,
                                              double new_time)
  {
    const double dt      = new_time - current_time;
    auto         do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_partitioner());
          new_position = part.get_position();
          new_position.add(dt, part.get_velocity());
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(penalty_parts, penalty_part_vectors);
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
    const double dt      = new_time - current_time;
    auto         do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_partitioner());
          new_position = part.get_position();
          new_position.add(dt, vectors.get_velocity(i, half_time));
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(penalty_parts, penalty_part_vectors);
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

    auto do_load =
      [&](auto &collection, auto &vectors, auto &forces, auto &right_hand_sides)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          const auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          forces.emplace_back(part.get_partitioner());
          right_hand_sides.emplace_back(part.get_partitioner());

          // The velocity isn't available at data_time so use current_time -
          // IBFEMethod does this too.
          const auto &position = vectors.get_position(i, data_time);
          const auto &velocity = vectors.get_velocity(i, current_time);
          // Unlike velocity interpolation and force spreading we actually need
          // the ghost values in the native partitioning, so make sure they are
          // available
          position.update_ghost_values();
          velocity.update_ghost_values();
          for (auto &force : part.get_force_contributions())
            force->setup_force(data_time, position, velocity);
          for (auto &active_strain : part.get_active_strains())
            active_strain->setup_strain(data_time);

          IBAMR_TIMER_START(t_compute_lagrangian_force_pk1);
          compute_load_vector(part.get_dof_handler(),
                              part.get_mapping(),
                              part.get_force_contributions(),
                              part.get_active_strains(),
                              data_time,
                              position,
                              velocity,
                              right_hand_sides[i]);
          IBAMR_TIMER_STOP(t_compute_lagrangian_force_pk1);
        }
    };
    std::deque<LinearAlgebra::distributed::Vector<double>> part_forces,
      part_right_hand_sides, penalty_part_forces, penalty_part_right_hand_sides;
    do_load(parts, part_vectors, part_forces, part_right_hand_sides);
    do_load(penalty_parts, penalty_part_vectors, penalty_part_forces, penalty_part_right_hand_sides);

    // Allow compression to overlap:
    auto do_compress = [](auto &vectors)
    {
      for (unsigned int i = 0; i < vectors.size(); ++i)
        vectors[i].compress_start(i, VectorOperation::add);
      for (unsigned int i = 0; i < vectors.size(); ++i)
        vectors[i].compress_finish(VectorOperation::add);
    };
    do_compress(part_right_hand_sides);
    do_compress(penalty_part_right_hand_sides);

    // And do the actual solve:
    auto do_solve = [&](const auto &collection,
                        const auto &interactions,
                        auto       &force_guesses,
                        auto       &vectors,
                        auto       &forces,
                        auto       &right_hand_sides)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          const auto &part = parts[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          if (interactions[i]->projection_is_interpolation())
            {
              vectors.set_force(i, data_time, std::move(right_hand_sides[i]));
            }
          else
            {
              IBAMR_TIMER_START(t_compute_lagrangian_force_solve);
              SolverControl control(
                input_db->getIntegerWithDefault("solver_iterations", 100),
                input_db->getDoubleWithDefault("solver_relative_tolerance",
                                               1e-6) *
                  right_hand_sides[i].l2_norm());
              SolverCG<LinearAlgebra::distributed::Vector<double>> cg(control);
              force_guesses[i].guess(forces[i], right_hand_sides[i]);
              cg.solve(part.get_mass_operator(),
                       forces[i],
                       right_hand_sides[i],
                       part.get_mass_preconditioner());
              force_guesses[i].submit(forces[i], right_hand_sides[i]);
              if (input_db->getBoolWithDefault("log_solver_iterations", false))
                {
                  tbox::plog << "IFEDMethod::computeLagrangianForce(): "
                             << "SolverCG<> converged in "
                             << control.last_step() << " steps." << std::endl;
                }
              vectors.set_force(i, data_time, std::move(forces[i]));
              IBAMR_TIMER_STOP(t_compute_lagrangian_force_solve);
            }
          for (auto &force : part.get_force_contributions())
            force->finish_force(data_time);
          for (auto &active_strain : part.get_active_strains())
            active_strain->finish_strain(data_time);
        }
    };
    do_solve(parts,
             interactions,
             force_guesses,
             part_vectors,
             part_forces,
             part_right_hand_sides);
    do_solve(penalty_parts,
             penalty_interactions,
             penalty_force_guesses,
             penalty_part_vectors,
             penalty_part_forces,
             penalty_part_right_hand_sides);
    IBAMR_TIMER_STOP(t_compute_lagrangian_force);
  }

  //
  // Data redistribution
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::reinit_interactions()
  {
    auto do_reinit = [&](const auto &collection, auto &interactions)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          const auto &part = collection[i];
          const auto &tria = dynamic_cast<
            const parallel::shared::Triangulation<structdim, spacedim> &>(
            part.get_triangulation());
          const auto &dof_handler = part.get_dof_handler();
          MappingFEField<structdim,
                         spacedim,
                         LinearAlgebra::distributed::Vector<double>>
            mapping(dof_handler, part.get_position());
          IBAMR_TIMER_START(t_reinit_interactions_bboxes);
          const auto local_bboxes =
            compute_cell_bboxes<structdim, spacedim, float>(dof_handler,
                                                            mapping);
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
          // We already check that this has a valid value earlier on
          const std::string interaction =
            input_db->getStringWithDefault("interaction", "ELEMENTAL");
          const int ln = primary_hierarchy->getFinestLevelNumber();

          tbox::Pointer<tbox::Database> interaction_db =
            new tbox::InputDatabase("interaction");
          // default database values are OK

          if (interaction == "ELEMENTAL")
            interactions[i]->reinit(interaction_db,
                                    tria,
                                    global_bboxes,
                                    global_edge_lengths,
                                    secondary_hierarchy.getSecondaryHierarchy(),
                                    std::make_pair(ln, ln));
          else
            {
              dynamic_cast<NodalInteraction<structdim, spacedim> &>(
                *interactions[i])
                .reinit(interaction_db,
                        tria,
                        global_bboxes,
                        secondary_hierarchy.getSecondaryHierarchy(),
                        std::make_pair(ln, ln),
                        part.get_dof_handler(),
                        part.get_position());
            }
          // TODO - we should probably add a reinit() function that sets up the
          // DoFHandler we always need
          interactions[i]->add_dof_handler(part.get_dof_handler());
        }
    };
    do_reinit(parts, interactions);
    do_reinit(penalty_parts, penalty_interactions);
    IBAMR_TIMER_STOP(t_reinit_interactions_objects);
  }


  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::beginDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_begin_data_redistribution);
    // This function is called before initializePatchHierarchy is - in that
    // case we don't have a hierarchy, so we don't have any data, and there is
    // naught to do
    if (primary_hierarchy)
      {
        // Weird things happen when we coarsen and refine if some levels are
        // not present, so fill them all in with zeros to start
        const int max_ln = primary_hierarchy->getFinestLevelNumber();
        fill_all(secondary_hierarchy.getSecondaryHierarchy(),
                 lagrangian_workload_current_index,
                 0,
                 max_ln);

        // Start:
        auto setup_transaction = [&](const auto &collection,
                                     const auto &interactions,
                                     auto       &transactions)
        {
          for (unsigned int i = 0; i < collection.size(); ++i)
            {
              const auto &part = collection[i];
              transactions.emplace_back(interactions[i]->add_workload_start(
                lagrangian_workload_current_index,
                part.get_position(),
                part.get_dof_handler()));
            }
        };
        std::vector<std::unique_ptr<TransactionBase>> transactions, penalty_transactions;
        setup_transaction(parts, interactions, transactions);
        setup_transaction(penalty_parts, penalty_interactions, penalty_transactions);

        // Compute:
        auto compute_transaction = [](auto &interactions, auto &transactions)
        {
          for (unsigned int i = 0; i < transactions.size(); ++i)
            transactions[i] = interactions[i]->add_workload_intermediate(
              std::move(transactions[i]));
        };
        compute_transaction(interactions, transactions);
        compute_transaction(penalty_interactions, penalty_transactions);

        // Finish:
        auto finish_transaction =
          [](const auto &interactions, auto &transactions)
        {
          for (unsigned int i = 0; i < transactions.size(); ++i)
            interactions[i]->add_workload_finish(std::move(transactions[i]));
        };
        finish_transaction(interactions, transactions);
        finish_transaction(penalty_interactions, penalty_transactions);

        // Move to primary hierarchy (we will read it back in
        // endDataRedistribution)
        fill_all(primary_hierarchy,
                 lagrangian_workload_current_index,
                 0,
                 max_ln);

        secondary_hierarchy.transferSecondaryToPrimary(
          max_ln,
          lagrangian_workload_current_index,
          lagrangian_workload_current_index,
          0.0);
      }

    // Clear a few things that depend on the current hierarchy:
    ghost_data_accumulator.reset();
    IBAMR_TIMER_STOP(t_begin_data_redistribution);
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::endDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    IBAMR_TIMER_START(t_end_data_redistribution);
    // same as beginDataRedistribution
    if (primary_hierarchy)
      {
        auto do_reset = [](auto &positions_regrid, const auto &collection)
        {
          positions_regrid.clear();
          for (unsigned int i = 0; i < collection.size(); ++i)
            positions_regrid.push_back(collection[i].get_position());
        };
        do_reset(positions_at_last_regrid, parts);
        do_reset(penalty_positions_at_last_regrid, penalty_parts);

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
        // timestepping. In particular: it will swap new and current,
        // deallocate new, reallocate new, and repeat. This causes (since we
        // only touch this variable in regrids) this data to get cleared, so
        // save a copy for plotting purposes.

        // TODO - implement a utility function for copying
        fill_all(primary_hierarchy,
                 lagrangian_workload_plot_index,
                 0,
                 primary_hierarchy->getFinestLevelNumber(),
                 0);
        extract_hierarchy_data_ops(lagrangian_workload_var, primary_hierarchy)
          ->copyData(lagrangian_workload_plot_index,
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
    auto do_put = [&](auto &collection, const std::string &prefix)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          std::ostringstream              out_str;
          boost::archive::binary_oarchive oarchive(out_str);
          collection[i].save(oarchive, 0);
          // TODO - with C++20 we can use view() instead of str() and skip
          // this copy
          const std::string out = out_str.str();
          save_binary(prefix + std::to_string(i),
                      out.c_str(),
                      out.c_str() + out.size(),
                      db);
        }
    };
    do_put(parts, "part_");
    do_put(penalty_parts, "penalty_part_");
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

  template class IFEDMethod<NDIM, NDIM>;
} // namespace fdl
