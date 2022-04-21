#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>

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

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase()
    : communicator(MPI_COMM_NULL)
    , level_number(std::numeric_limits<int>::max())
  {}

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float>                        &global_active_cell_lengths,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> p_hierarchy,
    const int                                         l_number)
    : communicator(MPI_COMM_NULL)
    , native_tria(&n_tria)
    , patch_hierarchy(p_hierarchy)
    , level_number(l_number)
  {
    reinit(n_tria,
           global_active_cell_bboxes,
           global_active_cell_lengths,
           p_hierarchy,
           l_number);
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &global_active_cell_bboxes,
    const std::vector<float> & /*global_active_cell_lengths*/,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> p_hierarchy,
    const int                                         l_number)
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
    level_number    = l_number;

    // Check inputs
    Assert(global_active_cell_bboxes.size() == native_tria->n_active_cells(),
           ExcMessage("There should be a bounding box for each active cell"));
    Assert(patch_hierarchy,
           ExcMessage("The provided pointer to a patch hierarchy should not be "
                      "null."));
    AssertIndexRange(l_number, patch_hierarchy->getNumberOfLevels());

    // Set up the patch map:
    {
      const auto patches =
        extract_patches(patch_hierarchy->getPatchLevel(level_number));
      // TODO we need to make extra ghost cell fraction a parameter
      const std::vector<BoundingBox<spacedim>> patch_bboxes =
        compute_patch_bboxes(patches, 1.0);
      BoxIntersectionPredicate<dim, spacedim> predicate(
        global_active_cell_bboxes, patch_bboxes, *native_tria);
      overlap_tria.reinit(*native_tria, predicate);

      // Yes, this is much more complex than necessary since
      // global_active_cell_bboxes is an argument to this function, but we don't
      // want to rely on that and p::s::T more than we have to since that
      // approach ultimately needs to go.
      //
      // TODO - we should refactor this into a more general function so we can
      // test it
      std::vector<CellId> bbox_cellids;
      for (const auto &cell : overlap_tria.active_cell_iterators())
        bbox_cellids.push_back(overlap_tria.get_native_cell_id(cell));

      // 1. Figure out who owns the bounding boxes we need:
      std::vector<types::subdomain_id> ranks =
        GridTools::get_subdomain_association(*native_tria, bbox_cellids);

      // 2. Send each processor the list of bboxes we need:
      std::map<types::subdomain_id,
               std::vector<std::pair<unsigned int, CellId>>>
        corresponding_requested_cellids;
      // Keep the overlap active cell index along for the ride
      for (unsigned int i = 0; i < ranks.size(); ++i)
        corresponding_requested_cellids[ranks[i]].emplace_back(i,
                                                               bbox_cellids[i]);

      const std::map<types::subdomain_id,
                     std::vector<std::pair<unsigned int, CellId>>>
        corresponding_cellids_to_send =
          Utilities::MPI::some_to_some(communicator,
                                       corresponding_requested_cellids);

      // 3. Send each processor the actual bboxes:
      std::map<
        types::subdomain_id,
        std::vector<std::pair<unsigned int, BoundingBox<spacedim, float>>>>
        requested_bboxes;
      for (const auto &pair : corresponding_cellids_to_send)
        {
          const auto  rank                = pair.first;
          const auto &indices_and_cellids = pair.second;

          auto &bboxes = requested_bboxes[rank];
          for (const auto &index_and_cellid : indices_and_cellids)
            {
              auto it =
                native_tria->create_cell_iterator(index_and_cellid.second);
              bboxes.emplace_back(
                index_and_cellid.first,
                global_active_cell_bboxes[it->active_cell_index()]);
            }
        }

      const auto received_bboxes =
        Utilities::MPI::some_to_some(communicator, requested_bboxes);

      std::vector<BoundingBox<spacedim, float>> overlap_bboxes(
        overlap_tria.n_active_cells());
      for (const auto &pair : received_bboxes)
        for (const auto &index_and_bbox : pair.second)
          {
            AssertIndexRange(index_and_bbox.first, overlap_bboxes.size());
            overlap_bboxes[index_and_bbox.first] = index_and_bbox.second;
          }

      // TODO add the ghost cell width as an input argument to this class
      patch_map.reinit(patches, 1.0, overlap_tria, overlap_bboxes);
    }

    // clear old dof info:
    native_dof_handlers.clear();
    overlap_dof_handlers.clear();
    overlap_to_native_dof_translations.clear();
    scatters.clear();
  }



  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::~InteractionBase()
  {
    int ierr = MPI_Comm_free(&communicator);
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
    this_dh_scatters.emplace_back(scatter);
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
  InteractionBase<dim, spacedim>::compute_projection_rhs_start(
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
    transaction.next_state = Transaction<dim, spacedim>::State::Intermediate;
    transaction.operation =
      Transaction<dim, spacedim>::Operation::Interpolation;

    transaction.position_scatter.global_to_overlap_start(
      *transaction.native_position, 0, transaction.overlap_position);

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

    trans.position_scatter.global_to_overlap_finish(*trans.native_position,
                                                    trans.overlap_position);

    // this is the point at which a base class would normally do computations.

    // After we compute we begin the scatter back to the native partitioning:

    // This object *cannot* get here without the first scatter finishing so
    // using channel 0 again is fine
    trans.rhs_scatter.overlap_to_global_start(trans.overlap_rhs,
                                              trans.rhs_scatter_back_op,
                                              0,
                                              *trans.native_rhs);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::compute_projection_rhs_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state == Transaction<dim, spacedim>::State::Finish),
           ExcMessage("Transaction state should be Finish"));

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
  InteractionBase<dim, spacedim>::compute_spread_start(
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
    transaction.next_state = Transaction<dim, spacedim>::State::Intermediate;
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

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

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
    Assert((trans.next_state == Transaction<dim, spacedim>::State::Finish),
           ExcMessage("Transaction state should be Finish"));

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
    (void)workload_index;
    (void)position;
    (void)position_dof_handler;

    return {};
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
    (void)t_ptr;
  }

  // instantiations

  template class InteractionBase<NDIM - 1, NDIM>;
  template class InteractionBase<NDIM, NDIM>;
} // namespace fdl
