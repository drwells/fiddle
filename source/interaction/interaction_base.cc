#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/intersection_predicate_lib.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/rtree.h>

#include <boost/container/small_vector.hpp>

#include <ibtk/IndexUtilities.h>
#include <ibtk/LEInteractor.h>

#include <memory>
#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  std::vector<MPI_Request>
  TransactionBase::delegate_outstanding_requests()
  {
    return {};
  }

  template <int dim, int spacedim>
  std::vector<MPI_Request>
  Transaction<dim, spacedim>::delegate_outstanding_requests()
  {
    auto copy1 = position_scatter.delegate_outstanding_requests();
    auto copy2 = solution_scatter.delegate_outstanding_requests();
    auto copy3 = rhs_scatter.delegate_outstanding_requests();

    std::vector<MPI_Request> result;
    result.insert(result.end(), copy1.begin(), copy1.end());
    result.insert(result.end(), copy2.begin(), copy2.end());
    result.insert(result.end(), copy3.begin(), copy3.end());
    return result;
  }

  template <int dim, int spacedim>
  std::vector<MPI_Request>
  WorkloadTransaction<dim, spacedim>::delegate_outstanding_requests()
  {
    return position_scatter.delegate_outstanding_requests();
  }

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase()
    : communicator(MPI_COMM_NULL)
    , level_numbers(
        {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()})
  {}

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float>                        &global_active_cell_lengths,
    tbox::Pointer<hier::PatchHierarchy<spacedim>>    p_hierarchy,
    const std::pair<int, int>                       &l_numbers)
    : communicator(MPI_COMM_NULL)
    , native_tria(&n_tria)
    , patch_hierarchy(p_hierarchy)
    , level_numbers(l_numbers)
  {
    reinit(input_db,
           n_tria,
           global_active_cell_bboxes,
           global_active_cell_lengths,
           p_hierarchy,
           l_numbers);
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::reinit(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float> & /*global_active_cell_lengths*/,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> p_hierarchy,
    const std::pair<int, int>                    &l_numbers)
  {
    // We don't need to create a communicator unless its the first time we are
    // here or if we, for some reason, get reinitialized with a totally new
    // Triangulation with a new network
    if (communicator == MPI_COMM_NULL ||
        native_tria->get_communicator() != n_tria.get_communicator())
      communicator =
        Utilities::MPI::duplicate_communicator(n_tria.get_communicator());

#ifdef DEBUG
    {
      int result = 0;
      int ierr   = MPI_Comm_compare(communicator,
                                  tbox::SAMRAI_MPI::getCommunicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT || result == MPI_IDENT,
             ExcMessage("The same communicator should be used for the "
                        "triangulation (from deal.II) and in SAMRAI"));
    }
#endif

    native_tria     = &n_tria;
    patch_hierarchy = p_hierarchy;
    level_numbers   = l_numbers;

    // Check inputs
    Assert(global_active_cell_bboxes.size() == native_tria->n_active_cells(),
           ExcMessage("There should be a bounding box for each active cell"));
    Assert(patch_hierarchy,
           ExcMessage("The provided pointer to a patch hierarchy should not be "
                      "null."));
    Assert(l_numbers.first <= l_numbers.second,
           ExcMessage("The coarser level number should be first"));
    AssertIndexRange(l_numbers.second, patch_hierarchy->getNumberOfLevels());

    // clear old dof info
    native_dof_handlers.clear();
    overlap_dof_handlers.clear();
    overlap_to_native_dof_translations.clear();
    scatters.clear();

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;
    for (int ln = level_numbers.first; ln <= level_numbers.second; ++ln)
      {
        const auto level_patches =
          extract_patches(patch_hierarchy->getPatchLevel(ln));
        patches.insert(patches.end(),
                       level_patches.begin(),
                       level_patches.end());
      }
    const std::vector<BoundingBox<spacedim>> patch_bboxes =
      compute_patch_bboxes(patches,
                           input_db->getDoubleWithDefault("ghost_cell_fraction",
                                                          1.0));
    BoxIntersectionPredicate<dim, spacedim> predicate(global_active_cell_bboxes,
                                                      patch_bboxes,
                                                      *native_tria);
    overlap_tria.reinit(*native_tria, predicate);
  }



  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::~InteractionBase()
  {
    int ierr = MPI_Comm_free(&communicator);
    (void)ierr;
    AssertNothrow(ierr == 0, ExcMessage("Unable to free the MPI communicator"));
  }



  template <int dim, int spacedim>
  DoFHandler<dim, spacedim> &
  InteractionBase<dim, spacedim>::get_overlap_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    auto iter = std::find(native_dof_handlers.begin(),
                          native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return *overlap_dof_handlers[iter - native_dof_handlers.begin()];
  }



  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  InteractionBase<dim, spacedim>::get_overlap_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler) const
  {
    auto iter = std::find(native_dof_handlers.begin(),
                          native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return *overlap_dof_handlers[iter - native_dof_handlers.begin()];
  }



  template <int dim, int spacedim>
  VectorOperation::values
  InteractionBase<dim, spacedim>::get_rhs_scatter_type() const
  {
    return VectorOperation::unknown;
  }



  template <int dim, int spacedim>
  Scatter<double>
  InteractionBase<dim, spacedim>::get_scatter(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    auto iter = std::find(native_dof_handlers.begin(),
                          native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));

    const std::size_t index = iter - native_dof_handlers.begin();
    if (index >= scatters.size())
      scatters.resize(index + 1);

    std::vector<Scatter<double>> &this_dh_scatters = scatters[index];
    if (this_dh_scatters.size() > 0)
      {
        Scatter<double> scatter = std::move(this_dh_scatters.back());
        this_dh_scatters.pop_back();
        return scatter;
      }

    Assert(index < overlap_to_native_dof_translations.size(),
           ExcFDLInternalError());
    Scatter<double> scatter(overlap_to_native_dof_translations[index],
                            native_dof_handler.locally_owned_dofs(),
                            communicator);
    return scatter;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::return_scatter(
    const DoFHandler<dim, spacedim> &native_dof_handler,
    Scatter<double>                &&scatter)
  {
    auto iter = std::find(native_dof_handlers.begin(),
                          native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));

    // TODO - add some more checking here to verify that the given scatter
    // corresponds to the given native_dof_handler
    const std::size_t index = iter - native_dof_handlers.begin();
    if (index >= scatters.size())
      scatters.resize(index + 1);
    std::vector<Scatter<double>> &this_dh_scatters = scatters[index];
    this_dh_scatters.emplace_back(std::move(scatter));
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::add_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    AssertThrow(&native_dof_handler.get_triangulation() == native_tria,
                ExcMessage("The DoFHandler must use the underlying native "
                           "triangulation."));
    const auto ptr = &native_dof_handler;
    if (std::find(native_dof_handlers.begin(),
                  native_dof_handlers.end(),
                  ptr) == native_dof_handlers.end())
      {
        native_dof_handlers.emplace_back(ptr);
        // TODO - implement a move ctor for DH in deal.II
        overlap_dof_handlers.emplace_back(
          std::make_unique<DoFHandler<dim, spacedim>>(overlap_tria));
        auto &overlap_dof_handler = *overlap_dof_handlers.back();
        overlap_dof_handler.distribute_dofs(
          native_dof_handler.get_fe_collection());

        std::vector<types::global_dof_index> overlap_to_native_dofs =
          compute_overlap_to_native_dof_translation(overlap_tria,
                                                    overlap_dof_handler,
                                                    native_dof_handler);
        overlap_to_native_dof_translations.emplace_back(
          std::move(overlap_to_native_dofs));
      }
  }



  template <int dim, int spacedim>
  bool
  InteractionBase<dim, spacedim>::projection_is_interpolation() const
  {
    return false;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_scatter_start(
    const std::string                                &kernel_name,
    const int                                         data_idx,
    const DoFHandler<dim, spacedim>                  &position_dof_handler,
    const LinearAlgebra::distributed::Vector<double> &position,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const Mapping<dim, spacedim>                     &mapping,
    LinearAlgebra::distributed::Vector<double>       &rhs)
  {
#ifdef DEBUG
    {
      int result = 0;
      int ierr   = MPI_Comm_compare(communicator,
                                  position.get_mpi_communicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage(
               "The same communicator should be used for position and the "
               "input triangulation"));
      ierr =
        MPI_Comm_compare(communicator, rhs.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for rhs and "
                        "the input triangulation"));
    }
#endif

    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.kernel_name      = kernel_name;
    transaction.current_data_idx = data_idx;

    // Setup position info:
    transaction.native_position_dof_handler = &position_dof_handler;
    transaction.native_position             = &position;
    transaction.overlap_position.reinit(
      get_overlap_dof_handler(position_dof_handler).n_dofs());
    transaction.position_scatter = get_scatter(position_dof_handler);

    // Setup rhs info:
    transaction.native_dof_handler = &dof_handler;
    transaction.mapping            = &mapping;
    transaction.native_rhs         = &rhs;
    transaction.overlap_rhs.reinit(
      get_overlap_dof_handler(dof_handler).n_dofs());
    transaction.rhs_scatter         = get_scatter(dof_handler);
    transaction.rhs_scatter_back_op = this->get_rhs_scatter_type();

    // Setup state:
    transaction.next_state = Transaction<dim, spacedim>::State::ScatterFinish;
    transaction.operation =
      Transaction<dim, spacedim>::Operation::Interpolation;

    transaction.position_scatter.global_to_overlap_start(
      *transaction.native_position, 0, transaction.overlap_position);

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_scatter_finish(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::ScatterFinish),
           ExcMessage("Transaction state should be ScatterFinish"));

    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    trans.next_state = Transaction<dim, spacedim>::State::Intermediate;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // this is the point at which a base class would normally do computations.

    trans.next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_accumulate_start(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::AccumulateStart),
           ExcMessage("Transaction state should be AccumulateStart"));

    // This object cannot get here without the first scatter finishing so using
    // channel 0 again is fine
    trans.rhs_scatter.overlap_to_global_start(trans.overlap_rhs,
                                              trans.rhs_scatter_back_op,
                                              0,
                                              *trans.native_rhs);

    trans.next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::compute_projection_rhs_accumulate_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    trans.rhs_scatter.overlap_to_global_finish(trans.overlap_rhs,
                                               trans.rhs_scatter_back_op,
                                               *trans.native_rhs);
    trans.next_state = Transaction<dim, spacedim>::State::Done;

    return_scatter(*trans.native_position_dof_handler,
                   std::move(trans.position_scatter));
    return_scatter(*trans.native_dof_handler, std::move(trans.rhs_scatter));
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_scatter_start(
    const std::string                                &kernel_name,
    const int                                         data_idx,
    const LinearAlgebra::distributed::Vector<double> &position,
    const DoFHandler<dim, spacedim>                  &position_dof_handler,
    const Mapping<dim, spacedim>                     &mapping,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const LinearAlgebra::distributed::Vector<double> &solution)
  {
#ifdef DEBUG
    {
      int result = 0;
      int ierr   = MPI_Comm_compare(communicator,
                                  position.get_mpi_communicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage(
               "The same communicator should be used for position and the "
               "input triangulation"));
      ierr = MPI_Comm_compare(communicator,
                              solution.get_mpi_communicator(),
                              &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage(
               "The same communicator should be used for solution and the "
               "input triangulation"));
    }
#endif

    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.kernel_name      = kernel_name;
    transaction.current_data_idx = data_idx;

    // Setup position info:
    transaction.native_position_dof_handler = &position_dof_handler;
    transaction.position_scatter            = get_scatter(position_dof_handler);
    transaction.native_position             = &position;
    transaction.overlap_position.reinit(
      get_overlap_dof_handler(position_dof_handler).n_dofs());

    // Setup solution info:
    transaction.native_dof_handler = &dof_handler;
    transaction.solution_scatter   = get_scatter(dof_handler);
    transaction.mapping            = &mapping;
    transaction.native_solution    = &solution;
    transaction.overlap_solution.reinit(
      get_overlap_dof_handler(dof_handler).n_dofs());

    // Setup state:
    transaction.next_state = Transaction<dim, spacedim>::State::ScatterFinish;
    transaction.operation  = Transaction<dim, spacedim>::Operation::Spreading;

    // OK, now start scattering:

    // Since we set up our own communicator in this object we can fearlessly use
    // channels 0 and 1 to guarantee traffic is not accidentally mingled
    transaction.position_scatter.global_to_overlap_start(
      *transaction.native_position, 0, transaction.overlap_position);

    transaction.solution_scatter.global_to_overlap_start(
      *transaction.native_solution, 1, transaction.overlap_solution);

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_scatter_finish(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::ScatterFinish),
           ExcMessage("Transaction state should be Intermediate"));

    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);
    trans.solution_scatter.global_to_overlap_finish(*trans.native_solution,
                                                    trans.overlap_solution);

    trans.next_state = Transaction<dim, spacedim>::State::Intermediate;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);
    trans.solution_scatter.global_to_overlap_finish(*trans.native_solution,
                                                    trans.overlap_solution);

    // this is the point at which a base class would normally do computations.

    trans.next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::compute_spread_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    // since no data is moved there is nothing else to do here

    trans.next_state = Transaction<dim, spacedim>::State::Done;

    return_scatter(*trans.native_position_dof_handler,
                   std::move(trans.position_scatter));
    return_scatter(*trans.native_dof_handler,
                   std::move(trans.solution_scatter));
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::add_workload_start(
    const int                                         workload_index,
    const LinearAlgebra::distributed::Vector<double> &position,
    const DoFHandler<dim, spacedim>                  &position_dof_handler)
  {
    auto t_ptr = std::make_unique<WorkloadTransaction<dim, spacedim>>();
    WorkloadTransaction<dim, spacedim> &transaction = *t_ptr;

    transaction.workload_index = workload_index;

    // Setup position info:
    transaction.native_position_dof_handler = &position_dof_handler;
    transaction.native_position             = &position;
    transaction.position_scatter = this->get_scatter(position_dof_handler);
    transaction.overlap_position.reinit(
      this->get_overlap_dof_handler(position_dof_handler).n_dofs());

    // Setup state:
    transaction.next_state =
      WorkloadTransaction<dim, spacedim>::State::Intermediate;

    transaction.position_scatter.global_to_overlap_start(
      *transaction.native_position, 0, transaction.overlap_position);

    return t_ptr;
  }

  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::add_workload_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    return t_ptr;
  }

  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::add_workload_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<WorkloadTransaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.next_state ==
            WorkloadTransaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    trans.next_state = WorkloadTransaction<dim, spacedim>::State::Done;

    this->return_scatter(*trans.native_position_dof_handler,
                         std::move(trans.position_scatter));
  }

  // instantiations

  template class InteractionBase<NDIM - 1, NDIM>;
  template class InteractionBase<NDIM, NDIM>;
} // namespace fdl
