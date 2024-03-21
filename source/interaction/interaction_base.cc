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
    auto copy1 = m_position_scatter.delegate_outstanding_requests();
    auto copy2 = m_solution_scatter.delegate_outstanding_requests();
    auto copy3 = m_rhs_scatter.delegate_outstanding_requests();

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
    return m_position_scatter.delegate_outstanding_requests();
  }

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase()
    : m_communicator(MPI_COMM_NULL)
    , m_level_numbers(
        {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()})
  {}

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float>                        &global_active_cell_lengths,
    tbox::Pointer<hier::PatchHierarchy<spacedim>>    patch_hierarchy,
    const std::pair<int, int>                       &level_numbers)
    : m_communicator(MPI_COMM_NULL)
    , m_native_tria(&n_tria)
    , m_patch_hierarchy(patch_hierarchy)
    , m_level_numbers(level_numbers)
  {
    reinit(input_db,
           n_tria,
           global_active_cell_bboxes,
           global_active_cell_lengths,
           m_patch_hierarchy,
           m_level_numbers);
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::reinit(
    const tbox::Pointer<tbox::Database>                  &input_db,
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float> & /*global_active_cell_lengths*/,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> p_hierarchy,
    const std::pair<int, int>                    &level_numbers)
  {
    // We don't need to create a communicator unless its the first time we are
    // here or if we, for some reason, get reinitialized with a totally new
    // Triangulation with a new network
    if (m_communicator == MPI_COMM_NULL ||
        m_native_tria->get_communicator() != n_tria.get_communicator())
      m_communicator =
        Utilities::MPI::duplicate_communicator(n_tria.get_communicator());

#ifdef DEBUG
    {
      int result = 0;
      int ierr   = MPI_Comm_compare(m_communicator,
                                  tbox::SAMRAI_MPI::getCommunicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT || result == MPI_IDENT,
             ExcMessage("The same communicator should be used for the "
                        "triangulation (from deal.II) and in SAMRAI"));
    }
#endif

    m_native_tria     = &n_tria;
    m_patch_hierarchy = p_hierarchy;
    m_level_numbers   = level_numbers;

    // Check inputs
    Assert(global_active_cell_bboxes.size() == m_native_tria->n_active_cells(),
           ExcMessage("There should be a bounding box for each active cell"));
    Assert(m_patch_hierarchy,
           ExcMessage("The provided pointer to a patch hierarchy should not be "
                      "null."));
    Assert(level_numbers.first <= level_numbers.second,
           ExcMessage("The coarser level number should be first"));
    AssertIndexRange(level_numbers.second, m_patch_hierarchy->getNumberOfLevels());

    // clear old dof info
    native_dof_handlers.clear();
    overlap_dof_handlers.clear();
    overlap_to_native_dof_translations.clear();
    scatters.clear();

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;
    for (int ln = m_level_numbers.first; ln <= m_level_numbers.second; ++ln)
      {
        const auto level_patches =
          extract_patches(m_patch_hierarchy->getPatchLevel(ln));
        patches.insert(patches.end(),
                       level_patches.begin(),
                       level_patches.end());
      }
    const std::vector<BoundingBox<spacedim, float>> patch_bboxes =
      compute_patch_bboxes<spacedim, float>(
        patches, input_db->getDoubleWithDefault("ghost_cell_fraction", 1.0));
    BoxIntersectionPredicate<dim, spacedim> predicate(global_active_cell_bboxes,
                                                      patch_bboxes,
                                                      *m_native_tria);
    m_overlap_tria.reinit(*m_native_tria, predicate);
  }



  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::~InteractionBase()
  {
    int ierr = MPI_Comm_free(&m_communicator);
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
                            m_communicator);
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
    AssertThrow(&native_dof_handler.get_triangulation() == m_native_tria,
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
          std::make_unique<DoFHandler<dim, spacedim>>(m_overlap_tria));
        auto &overlap_dof_handler = *overlap_dof_handlers.back();
        overlap_dof_handler.distribute_dofs(
          native_dof_handler.get_fe_collection());

        std::vector<types::global_dof_index> overlap_to_native_dofs =
          compute_overlap_to_native_dof_translation(m_overlap_tria,
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
      int ierr   = MPI_Comm_compare(m_communicator,
                                  position.get_mpi_communicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage(
               "The same communicator should be used for position and the "
               "input triangulation"));
      ierr =
        MPI_Comm_compare(m_communicator, rhs.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for rhs and "
                        "the input triangulation"));
    }
#endif

    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.m_kernel_name      = kernel_name;
    transaction.m_current_data_idx = data_idx;

    // Setup position info:
    transaction.m_native_position_dof_handler = &position_dof_handler;
    transaction.m_native_position             = &position;
    transaction.m_overlap_position.reinit(
      get_overlap_dof_handler(position_dof_handler).n_dofs());
    transaction.m_position_scatter = get_scatter(position_dof_handler);

    // Setup rhs info:
    transaction.m_native_dof_handler = &dof_handler;
    transaction.m_mapping            = &mapping;
    transaction.m_native_rhs         = &rhs;
    transaction.m_overlap_rhs.reinit(
      get_overlap_dof_handler(dof_handler).n_dofs());
    transaction.m_rhs_scatter         = get_scatter(dof_handler);
    transaction.m_rhs_scatter_back_op = this->get_rhs_scatter_type();

    // Setup state:
    transaction.m_next_state = Transaction<dim, spacedim>::State::ScatterFinish;
    transaction.m_operation =
      Transaction<dim, spacedim>::Operation::Interpolation;

    transaction.m_position_scatter.global_to_overlap_start(
      *transaction.m_native_position, 0, transaction.m_overlap_position);

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_scatter_finish(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::ScatterFinish),
           ExcMessage("Transaction state should be ScatterFinish"));

    trans.m_position_scatter.global_to_overlap_finish(*trans.m_native_position,
                                                      trans.m_overlap_position);

    trans.m_next_state = Transaction<dim, spacedim>::State::Intermediate;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    // this is the point at which a base class would normally do computations.

    trans.m_next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_accumulate_start(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::AccumulateStart),
           ExcMessage("Transaction state should be AccumulateStart"));

    // This object cannot get here without the first scatter finishing so using
    // channel 0 again is fine
    trans.m_rhs_scatter.overlap_to_global_start(trans.m_overlap_rhs,
                                                trans.m_rhs_scatter_back_op,
                                                0,
                                                *trans.m_native_rhs);

    trans.m_next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::compute_projection_rhs_accumulate_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    trans.m_rhs_scatter.overlap_to_global_finish(trans.m_overlap_rhs,
                                                 trans.m_rhs_scatter_back_op,
                                                 *trans.m_native_rhs);
    trans.m_next_state = Transaction<dim, spacedim>::State::Done;

    return_scatter(*trans.m_native_position_dof_handler,
                   std::move(trans.m_position_scatter));
    return_scatter(*trans.m_native_dof_handler, std::move(trans.m_rhs_scatter));
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
      int ierr   = MPI_Comm_compare(m_communicator,
                                  position.get_mpi_communicator(),
                                  &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage(
               "The same communicator should be used for position and the "
               "input triangulation"));
      ierr = MPI_Comm_compare(m_communicator,
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
    transaction.m_kernel_name      = kernel_name;
    transaction.m_current_data_idx = data_idx;

    // Setup position info:
    transaction.m_native_position_dof_handler = &position_dof_handler;
    transaction.m_position_scatter = get_scatter(position_dof_handler);
    transaction.m_native_position  = &position;
    transaction.m_overlap_position.reinit(
      get_overlap_dof_handler(position_dof_handler).n_dofs());

    // Setup solution info:
    transaction.m_native_dof_handler = &dof_handler;
    transaction.m_solution_scatter   = get_scatter(dof_handler);
    transaction.m_mapping            = &mapping;
    transaction.m_native_solution    = &solution;
    transaction.m_overlap_solution.reinit(
      get_overlap_dof_handler(dof_handler).n_dofs());

    // Setup state:
    transaction.m_next_state = Transaction<dim, spacedim>::State::ScatterFinish;
    transaction.m_operation  = Transaction<dim, spacedim>::Operation::Spreading;

    // OK, now start scattering:

    // Since we set up our own communicator in this object we can fearlessly use
    // channels 0 and 1 to guarantee traffic is not accidentally mingled
    transaction.m_position_scatter.global_to_overlap_start(
      *transaction.m_native_position, 0, transaction.m_overlap_position);

    transaction.m_solution_scatter.global_to_overlap_start(
      *transaction.m_native_solution, 1, transaction.m_overlap_solution);

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_scatter_finish(
    std::unique_ptr<TransactionBase> t_ptr) const
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::ScatterFinish),
           ExcMessage("Transaction state should be Intermediate"));

    trans.m_position_scatter.global_to_overlap_finish(*trans.m_native_position,
                                                      trans.m_overlap_position);
    trans.m_solution_scatter.global_to_overlap_finish(*trans.m_native_solution,
                                                      trans.m_overlap_solution);

    trans.m_next_state = Transaction<dim, spacedim>::State::Intermediate;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    trans.m_position_scatter.global_to_overlap_finish(*trans.m_native_position,
                                                      trans.m_overlap_position);
    trans.m_solution_scatter.global_to_overlap_finish(*trans.m_native_solution,
                                                      trans.m_overlap_solution);

    // this is the point at which a base class would normally do computations.

    trans.m_next_state = Transaction<dim, spacedim>::State::AccumulateFinish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::compute_spread_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.m_operation ==
            Transaction<dim, spacedim>::Operation::Spreading),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.m_next_state ==
            Transaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    // since no data is moved there is nothing else to do here

    trans.m_next_state = Transaction<dim, spacedim>::State::Done;

    return_scatter(*trans.m_native_position_dof_handler,
                   std::move(trans.m_position_scatter));
    return_scatter(*trans.m_native_dof_handler,
                   std::move(trans.m_solution_scatter));
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

    transaction.m_workload_index = workload_index;

    // Setup position info:
    transaction.m_native_position_dof_handler = &position_dof_handler;
    transaction.m_native_position             = &position;
    transaction.m_position_scatter = this->get_scatter(position_dof_handler);
    transaction.m_overlap_position.reinit(
      this->get_overlap_dof_handler(position_dof_handler).n_dofs());

    // Setup state:
    transaction.m_next_state =
      WorkloadTransaction<dim, spacedim>::State::Intermediate;

    transaction.m_position_scatter.global_to_overlap_start(
      *transaction.m_native_position, 0, transaction.m_overlap_position);

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
    Assert((trans.m_next_state ==
            WorkloadTransaction<dim, spacedim>::State::AccumulateFinish),
           ExcMessage("Transaction state should be AccumulateFinish"));

    trans.m_next_state = WorkloadTransaction<dim, spacedim>::State::Done;

    this->return_scatter(*trans.m_native_position_dof_handler,
                         std::move(trans.m_position_scatter));
  }

  // instantiations

  template class InteractionBase<NDIM - 1, NDIM>;
  template class InteractionBase<NDIM, NDIM>;
} // namespace fdl
