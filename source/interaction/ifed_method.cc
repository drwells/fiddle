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
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/lac/solver_cg.h>

#include <ibamr/IBHierarchyIntegrator.h>
#include <ibamr/ibamr_utilities.h>

#include <ibtk/IBTK_MPI.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/RobinPhysBdryPatchStrategy.h>
#include <ibtk/ibtk_utilities.h>

#include <CellVariable.h>
#include <HierarchyDataOpsManager.h>
#include <IntVector.h>
#include <VariableDatabase.h>
#include <tbox/TimerManager.h>

#include <deque>

namespace
{
  using namespace SAMRAI;
  static tbox::Timer *t_interpolate_velocity;
  static tbox::Timer *t_interpolate_velocity_start_barrier;
  static tbox::Timer *t_interpolate_velocity_rhs;
  static tbox::Timer *t_interpolate_velocity_solve;
  static tbox::Timer *t_interpolate_velocity_solve_start_barrier;
  static tbox::Timer *t_compute_lagrangian_force;
  static tbox::Timer *t_compute_lagrangian_force_start_barrier;
  static tbox::Timer *t_compute_lagrangian_force_position_ghost_update;
  static tbox::Timer *t_compute_lagrangian_force_setup_force_and_strain;
  static tbox::Timer *t_compute_lagrangian_force_pk1;
  static tbox::Timer *t_compute_lagrangian_force_pre_compress_barrier;
  static tbox::Timer *t_compute_lagrangian_force_compress_vector;
  static tbox::Timer *t_compute_lagrangian_force_solve;
  static tbox::Timer *t_spread_force;
  static tbox::Timer *t_spread_force_start_barrier;
  static tbox::Timer *t_compute_lagrangian_fluid_source;
  static tbox::Timer *t_spread_fluid_source;
  static tbox::Timer *t_add_workload_estimate;
  static tbox::Timer *t_begin_data_redistribution;
  static tbox::Timer *t_end_data_redistribution;
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
    : IFEDMethod<dim, spacedim>(object_name,
                                input_input_db,
                                {},
                                std::move(input_parts),
                                register_for_restart)
  {}

  template <int dim, int spacedim>
  IFEDMethod<dim, spacedim>::IFEDMethod(
    const std::string                     &object_name,
    tbox::Pointer<tbox::Database>          input_input_db,
    std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
    std::vector<Part<dim, spacedim>>     &&input_parts,
    const bool                             register_for_restart)
    : IFEDMethodBase<dim, spacedim>(object_name,
                                    std::move(input_surface_parts),
                                    std::move(input_parts),
                                    register_for_restart)
    , input_db(copy_database(input_input_db))
    , ghosts(0)
    , secondary_hierarchy(object_name + "::secondary_hierarchy",
                          input_db->getDatabase("GriddingAlgorithm"),
                          input_db->getDatabase("LoadBalancer"))
  {
    const std::string interaction =
      input_db->getStringWithDefault("interaction", "ELEMENTAL");
    if (interaction == "ELEMENTAL")
      {
        AssertThrow(this->n_surface_parts() == 0, ExcFDLNotImplemented());
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
        init_elemental(interactions,
                       force_guesses,
                       velocity_guesses,
                       this->parts);
        init_elemental(surface_interactions,
                       surface_force_guesses,
                       surface_velocity_guesses,
                       this->surface_parts);
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
        init_nodal(interactions, this->parts);
        init_nodal(surface_interactions, this->surface_parts);
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("unsupported interaction type " + interaction +
                               "."));
      }

    auto do_kernel = [&](const std::string        &key,
                         const auto               &collection,
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
                                 "either be 1 or equal the number of (surface) "
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
    do_kernel("IB_kernel", this->parts, ib_kernels);
    do_kernel("surface_IB_kernel", this->surface_parts, surface_ib_kernels);

    auto set_timer = [&](const char *name)
    { return tbox::TimerManager::getManager()->getTimer(name); };

    t_interpolate_velocity =
      set_timer("fdl::IFEDMethod::interpolateVelocity()");
    t_interpolate_velocity_start_barrier =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[start_barrier]");
    t_interpolate_velocity_rhs =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[rhs]");
    t_interpolate_velocity_solve =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[solve]");
    t_interpolate_velocity_solve_start_barrier =
      set_timer("fdl::IFEDMethod::interpolateVelocity()[solve_start_barrier]");
    t_compute_lagrangian_force =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()");
    t_compute_lagrangian_force_start_barrier =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[start_barrier]");
    t_compute_lagrangian_force_position_ghost_update = set_timer(
      "fdl::IFEDMethod::computeLagrangianForce()[position_ghost_update]");
    t_compute_lagrangian_force_setup_force_and_strain = set_timer(
      "fdl::IFEDMethod::computeLagrangianForce()[setup_force_and_strain]");
    t_compute_lagrangian_force_pk1 =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[pk1]");
    t_compute_lagrangian_force_pre_compress_barrier = set_timer(
      "fdl::IFEDMethod::computeLagrangianForce()[pre_compress_barrier]");
    t_compute_lagrangian_force_compress_vector =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[compress_vector]");
    t_compute_lagrangian_force_solve =
      set_timer("fdl::IFEDMethod::computeLagrangianForce()[solve]");
    t_spread_force = set_timer("fdl::IFEDMethod::spreadForce()");
    t_spread_force_start_barrier =
      set_timer("fdl::IFEDMethod::spreadForce()[start_barrier]");
    t_compute_lagrangian_fluid_source =
      set_timer("fdl::IFEDMethod::computeLagrangianFluidSource()");
    t_spread_fluid_source = set_timer("fdl::IFEDMethod::spreadFluidSource()");
    t_add_workload_estimate =
      set_timer("fdl::IFEDMethod::addWorkloadEstimate()");
    t_begin_data_redistribution =
      set_timer("fdl::IFEDMethod::beginDataRedistribution()");
    t_end_data_redistribution =
      set_timer("fdl::IFEDMethod::endDataRedistribution()");
    t_reinit_interactions_bboxes =
      set_timer("fdl::IFEDMethod::reinit_interactions()[bboxes]");
    t_reinit_interactions_edges =
      set_timer("fdl::IFEDMethod::reinit_interactions()[edges]");
    t_reinit_interactions_objects =
      set_timer("fdl::IFEDMethod::reinit_interactions()[objects]");
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::initializePatchHierarchy(
    tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg,
    int                                              u_data_index,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      &u_synch_scheds,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
          &u_ghost_fill_scheds,
    int    integrator_step,
    double init_data_time,
    bool   initial_time)
  {
    IFEDMethodBase<dim, spacedim>::initializePatchHierarchy(hierarchy,
                                                            gridding_alg,
                                                            u_data_index,
                                                            u_synch_scheds,
                                                            u_ghost_fill_scheds,
                                                            integrator_step,
                                                            init_data_time,
                                                            initial_time);

    secondary_hierarchy.reinit(this->patch_hierarchy->getFinestLevelNumber(),
                               this->patch_hierarchy->getFinestLevelNumber(),
                               this->patch_hierarchy);

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
      &/*u_synch_scheds*/,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
          &/*u_ghost_fill_scheds*/,
    double data_time)
  {
#ifdef FDL_ENABLE_TIMER_BARRIERS
    {
      ScopedTimer t0(t_interpolate_velocity_start_barrier);
      const int ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
      AssertThrowMPI(ierr);
    }
#endif
    ScopedTimer t1(t_interpolate_velocity);

    // Update the secondary hierarchy:
    secondary_hierarchy.transferPrimaryToSecondary(
      this->patch_hierarchy->getFinestLevelNumber(),
      u_data_index,
      u_data_index,
      data_time,
      this->d_ib_solver->getVelocityPhysBdryOp());

    IBAMR_TIMER_START(t_interpolate_velocity_rhs);
    std::vector<MPI_Request> requests;
    // native to overlap:
    auto scatter_start = [&](const auto &collection,
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
            interactions[i]->compute_projection_rhs_scatter_start(
              kernels[i],
              u_data_index,
              part.get_dof_handler(),
              vectors.get_position(i, data_time),
              part.get_dof_handler(),
              part.get_mapping(),
              rhs_vectors[i]));
          auto current_requests =
            transactions[i]->delegate_outstanding_requests();
          requests.insert(requests.end(),
                          current_requests.begin(),
                          current_requests.end());
        }
    };
    auto scatter_finish = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        transactions[i] =
          interactions[i]->compute_projection_rhs_scatter_finish(
            std::move(transactions[i]));
    };
    // we emplace_back so use a deque to keep pointers valid
    std::vector<std::unique_ptr<TransactionBase>> transactions,
      surface_transactions;
    std::deque<LinearAlgebra::distributed::Vector<double>> rhs_vecs,
      surface_rhs_vecs;
    scatter_start(this->parts,
                  interactions,
                  ib_kernels,
                  this->part_vectors,
                  transactions,
                  rhs_vecs);
    scatter_start(this->surface_parts,
                  surface_interactions,
                  surface_ib_kernels,
                  this->surface_part_vectors,
                  surface_transactions,
                  surface_rhs_vecs);
    int ierr =
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
    requests.resize(0);
    scatter_finish(interactions, transactions);
    scatter_finish(surface_interactions, surface_transactions);

    // Compute:
    auto compute_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        transactions[i] = interactions[i]->compute_projection_rhs_intermediate(
          std::move(transactions[i]));
    };
    compute_transaction(interactions, transactions);
    compute_transaction(surface_interactions, surface_transactions);

    // Move back:
    auto accumulate_start = [&](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        {
          transactions[i] =
            interactions[i]->compute_projection_rhs_accumulate_start(
              std::move(transactions[i]));
          auto current_requests =
            transactions[i]->delegate_outstanding_requests();
          requests.insert(requests.end(),
                          current_requests.begin(),
                          current_requests.end());
        }
    };
    auto accumulate_finish = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        interactions[i]->compute_projection_rhs_accumulate_finish(
          std::move(transactions[i]));
    };
    accumulate_start(interactions, transactions);
    accumulate_start(surface_interactions, surface_transactions);
    ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
    requests.resize(0);
    accumulate_finish(interactions, transactions);
    accumulate_finish(surface_interactions, surface_transactions);

    IBAMR_TIMER_STOP(t_interpolate_velocity_rhs);
    // We cannot start the linear solve without first finishing this, so use a
    // barrier to keep the timers accurate
#ifdef FDL_ENABLE_TIMER_BARRIERS
    {
      ScopedTimer t2(t_interpolate_velocity_solve_start_barrier)
      ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
      AssertThrowMPI(ierr);
    }
#endif

    // Project:
    ScopedTimer t3(t_interpolate_velocity_solve);
    auto do_solve = [&](const auto &collection,
                        const auto &interactions,
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
    do_solve(this->parts,
             interactions,
             this->part_vectors,
             velocity_guesses,
             rhs_vecs);
    do_solve(this->surface_parts,
             surface_interactions,
             this->surface_part_vectors,
             surface_velocity_guesses,
             surface_rhs_vecs);
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
#ifdef FDL_ENABLE_TIMER_BARRIERS
    {
      ScopedTimer t0(t_spread_force_start_barrier);
      const int ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
      AssertThrowMPI(ierr);
    }
#endif
    ScopedTimer t1(t_spread_force);
    const int level_number = this->patch_hierarchy->getFinestLevelNumber();

    std::shared_ptr<IBTK::SAMRAIDataCache> data_cache =
      secondary_hierarchy.getSAMRAIDataCache();
    auto       hierarchy = secondary_hierarchy.getSecondaryHierarchy();
    const auto f_scratch_data_index =
      data_cache->getCachedPatchDataIndex(f_data_index);
    fill_all(hierarchy, f_scratch_data_index, level_number, level_number, 0.0);

    std::vector<MPI_Request> requests;
    // native to overlap:
    auto scatter_start = [&](const auto &collection,
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
          transactions.emplace_back(
            interactions[i]->compute_spread_scatter_start(
              kernels[i],
              f_scratch_data_index,
              vectors.get_position(i, data_time),
              part.get_dof_handler(),
              part.get_mapping(),
              part.get_dof_handler(),
              vectors.get_force(i, data_time)));
          auto current_requests =
            transactions[i]->delegate_outstanding_requests();
          requests.insert(requests.end(),
                          current_requests.begin(),
                          current_requests.end());
        }
    };
    auto scatter_finish = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        transactions[i] = interactions[i]->compute_spread_scatter_finish(
          std::move(transactions[i]));
    };
    std::vector<std::unique_ptr<TransactionBase>> transactions,
      surface_transactions;
    scatter_start(
      this->parts, interactions, ib_kernels, this->part_vectors, transactions);
    scatter_start(this->surface_parts,
                  surface_interactions,
                  surface_ib_kernels,
                  this->surface_part_vectors,
                  surface_transactions);
    int ierr =
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
    requests.resize(0);
    scatter_finish(interactions, transactions);
    scatter_finish(surface_interactions, surface_transactions);

    // Compute:
    auto compute_transaction = [&](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        {
          transactions[i] = interactions[i]->compute_spread_intermediate(
            std::move(transactions[i]));
          auto current_requests =
            transactions[i]->delegate_outstanding_requests();
          requests.insert(requests.end(),
                          current_requests.begin(),
                          current_requests.end());
        }
    };
    compute_transaction(interactions, transactions);
    compute_transaction(surface_interactions, surface_transactions);
    ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    AssertThrowMPI(ierr);
    requests.resize(0);

    // Collect:
    auto collect_transaction = [](const auto &interactions, auto &transactions)
    {
      for (unsigned int i = 0; i < interactions.size(); ++i)
        interactions[i]->compute_spread_finish(std::move(transactions[i]));
    };
    collect_transaction(interactions, transactions);
    collect_transaction(surface_interactions, surface_transactions);

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
        extract_hierarchy_data_ops(f_var, this->patch_hierarchy);
      f_primary_data_ops->resetLevels(level_number, level_number);
      const auto f_primary_scratch_data_index =
        this->eulerian_data_cache->getCachedPatchDataIndex(f_data_index);
      // we have to zero everything here since the scratch to primary
      // communication does not touch ghost cells, which may have junk
      fill_all(this->patch_hierarchy,
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
  }

  //
  // Mechanics
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::computeLagrangianForce(double data_time)
  {
#ifdef FDL_ENABLE_TIMER_BARRIERS
    {
      ScopedTimer t0(t_compute_lagrangian_force_start_barrier);
      const int ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
      AssertThrowMPI(ierr);
    }
#endif
    ScopedTimer t1(t_compute_lagrangian_force);

    unsigned int channel = 0;
    auto         do_load =
      [&](auto &collection, auto &vectors, auto &forces, auto &right_hand_sides)
    {
      // Unlike velocity interpolation and force spreading we actually need
      // the ghost values in the native partitioning, so make sure they are
      // available
      IBAMR_TIMER_START(t_compute_lagrangian_force_position_ghost_update);
      for (unsigned int i = 0; i < collection.size(); ++i)
        vectors.get_position(i, data_time).update_ghost_values_start(channel++);
      for (unsigned int i = 0; i < collection.size(); ++i)
        vectors.get_position(i, data_time).update_ghost_values_finish();
      IBAMR_TIMER_STOP(t_compute_lagrangian_force_position_ghost_update);

      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          const auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          forces.emplace_back(part.get_partitioner());
          right_hand_sides.emplace_back(part.get_partitioner());

          const auto &position = vectors.get_position(i, data_time);
          // The velocity isn't available at data_time so use current_time -
          // IBFEMethod does this too. This is always equal to the part's
          // current velocity and, by convention, has up-to-date ghost values.
          const auto &velocity = part.get_velocity();
          IBAMR_TIMER_START(t_compute_lagrangian_force_setup_force_and_strain);
          for (auto &force : part.get_force_contributions())
            force->setup_force(data_time, position, velocity);
          for (auto &active_strain : part.get_active_strains())
            active_strain->setup_strain(data_time, position, velocity);
          IBAMR_TIMER_STOP(t_compute_lagrangian_force_setup_force_and_strain);

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
      part_right_hand_sides, surface_part_forces, surface_part_right_hand_sides;
    do_load(this->parts,
            this->part_vectors,
            part_forces,
            part_right_hand_sides);
    do_load(this->surface_parts,
            this->surface_part_vectors,
            surface_part_forces,
            surface_part_right_hand_sides);

#ifdef FDL_ENABLE_TIMER_BARRIERS
    {
      ScopedTimer t2(t_compute_lagrangian_force_pre_compress_barrier);
      const int ierr = MPI_Barrier(IBTK::IBTK_MPI::getCommunicator());
      AssertThrowMPI(ierr);
    }
#endif
    // Allow compression to overlap:
    auto do_compress = [](auto &vectors)
    {
      ScopedTimer t3(t_compute_lagrangian_force_compress_vector);
      for (unsigned int i = 0; i < vectors.size(); ++i)
        vectors[i].compress_start(i, VectorOperation::add);
      for (unsigned int i = 0; i < vectors.size(); ++i)
        vectors[i].compress_finish(VectorOperation::add);
    };
    do_compress(part_right_hand_sides);
    do_compress(surface_part_right_hand_sides);

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
          const auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          if (interactions[i]->projection_is_interpolation())
            {
              vectors.set_force(i, data_time, std::move(right_hand_sides[i]));
            }
          else
            {
              ScopedTimer t4(t_compute_lagrangian_force_solve);
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
            }
          for (auto &force : part.get_force_contributions())
            force->finish_force(data_time);
          for (auto &active_strain : part.get_active_strains())
            active_strain->finish_strain(data_time);
        }
    };
    do_solve(this->parts,
             interactions,
             force_guesses,
             this->part_vectors,
             part_forces,
             part_right_hand_sides);
    do_solve(this->surface_parts,
             surface_interactions,
             surface_force_guesses,
             this->surface_part_vectors,
             surface_part_forces,
             surface_part_right_hand_sides);
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
          const int ln = this->patch_hierarchy->getFinestLevelNumber();

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
          IBAMR_TIMER_STOP(t_reinit_interactions_objects);
        }
    };
    do_reinit(this->parts, interactions);
    do_reinit(this->surface_parts, surface_interactions);
  }


  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::beginDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg)
  {
    ScopedTimer t0(t_begin_data_redistribution);
    IFEDMethodBase<dim, spacedim>::beginDataRedistribution(hierarchy,
                                                           gridding_alg);
    // This function is called before initializePatchHierarchy is - in that
    // case we don't have a hierarchy, so we don't have any data, and there is
    // naught to do
    if (this->patch_hierarchy)
      {
        // Weird things happen when we coarsen and refine if some levels are
        // not present, so fill them all in with zeros to start
        const int max_ln = this->patch_hierarchy->getFinestLevelNumber();
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
        std::vector<std::unique_ptr<TransactionBase>> transactions,
          surface_transactions;
        setup_transaction(this->parts, interactions, transactions);
        setup_transaction(this->surface_parts,
                          surface_interactions,
                          surface_transactions);

        // Compute:
        auto compute_transaction = [](auto &interactions, auto &transactions)
        {
          for (unsigned int i = 0; i < transactions.size(); ++i)
            transactions[i] = interactions[i]->add_workload_intermediate(
              std::move(transactions[i]));
        };
        compute_transaction(interactions, transactions);
        compute_transaction(surface_interactions, surface_transactions);

        // Finish:
        auto finish_transaction =
          [](const auto &interactions, auto &transactions)
        {
          for (unsigned int i = 0; i < transactions.size(); ++i)
            interactions[i]->add_workload_finish(std::move(transactions[i]));
        };
        finish_transaction(interactions, transactions);
        finish_transaction(surface_interactions, surface_transactions);

        // Move to primary hierarchy (we will read it back in
        // endDataRedistribution)
        fill_all(this->patch_hierarchy,
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
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::endDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>>    hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> gridding_alg)
  {
    ScopedTimer t0(t_end_data_redistribution);
    // same as beginDataRedistribution
    if (this->patch_hierarchy)
      {
        secondary_hierarchy.reinit(
          this->patch_hierarchy->getFinestLevelNumber(),
          this->patch_hierarchy->getFinestLevelNumber(),
          this->patch_hierarchy,
          lagrangian_workload_current_index);

        reinit_interactions();

        if (input_db->getBoolWithDefault("enable_logging", true) &&
            (this->started_time_integration ||
             (!this->started_time_integration &&
              !input_db->getBoolWithDefault("skip_initial_workload", false))))
          {
            const int ln = this->patch_hierarchy->getFinestLevelNumber();
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
        fill_all(this->patch_hierarchy,
                 lagrangian_workload_plot_index,
                 0,
                 this->patch_hierarchy->getFinestLevelNumber(),
                 0);
        extract_hierarchy_data_ops(lagrangian_workload_var,
                                   this->patch_hierarchy)
          ->copyData(lagrangian_workload_plot_index,
                     lagrangian_workload_current_index,
                     false);
      }
    IFEDMethodBase<dim, spacedim>::endDataRedistribution(hierarchy,
                                                         gridding_alg);
  }

  //
  // Book-keeping
  //

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::registerEulerianVariables()
  {
    // we need ghosts for CONSERVATIVE_LINEAR_REFINE
    const hier::IntVector<spacedim> ghosts = 1;
    lagrangian_workload_var = new pdat::CellVariable<spacedim, double>(
      this->object_name + "::lagrangian_workload");
    this->registerVariable(lagrangian_workload_current_index,
                           lagrangian_workload_new_index,
                           lagrangian_workload_scratch_index,
                           lagrangian_workload_var,
                           ghosts,
                           "CONSERVATIVE_COARSEN",
                           "CONSERVATIVE_LINEAR_REFINE");

    auto *var_db  = hier::VariableDatabase<spacedim>::getDatabase();
    auto  context = var_db->getContext(this->object_name);
    lagrangian_workload_plot_index =
      var_db->registerVariableAndContext(lagrangian_workload_var,
                                         context,
                                         ghosts);
  }

  template class IFEDMethod<NDIM, NDIM>;
} // namespace fdl
