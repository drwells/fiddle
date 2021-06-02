#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.templates.h>

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

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase()
    : communicator(MPI_COMM_NULL)
    , level_number(std::numeric_limits<int>::max())
  {}

  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> & global_active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> p_hierarchy,
    const int                                         l_number)
    : communicator(MPI_COMM_NULL)
    , native_tria(&n_tria)
    , patch_hierarchy(p_hierarchy)
    , level_number(l_number)
  {
    reinit(n_tria, global_active_cell_bboxes, p_hierarchy, l_number);
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> & global_active_cell_bboxes,
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

      std::vector<BoundingBox<spacedim, float>> overlap_bboxes;
      for (const auto &cell : overlap_tria.active_cell_iterators())
        {
          auto native_cell = overlap_tria.get_native_cell(cell);
          overlap_bboxes.push_back(
            global_active_cell_bboxes[native_cell->active_cell_index()]);
        }

      // TODO add the ghost cell width as an input argument to this class
      patch_map.reinit(patches, 1.0, overlap_tria, overlap_bboxes);
    }
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
  Scatter<double> &
  InteractionBase<dim, spacedim>::get_scatter(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    auto iter = std::find(native_dof_handlers.begin(),
                          native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return scatters[iter - native_dof_handlers.begin()];
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

        const std::vector<types::global_dof_index> overlap_to_native_dofs =
          compute_overlap_to_native_dof_translation(overlap_tria,
                                                    overlap_dof_handler,
                                                    native_dof_handler);
        scatters.emplace_back(overlap_to_native_dofs,
                              native_dof_handler.locally_owned_dofs(),
                              communicator);
      }
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_start(
    const int                                         f_data_idx,
    const DoFHandler<dim, spacedim> &                 X_dof_handler,
    const LinearAlgebra::distributed::Vector<double> &X,
    const DoFHandler<dim, spacedim> &                 F_dof_handler,
    const Mapping<dim, spacedim> &                    F_mapping,
    LinearAlgebra::distributed::Vector<double> &      F_rhs)
  {
#ifdef DEBUG
    {
      int result = 0;
      int ierr =
        MPI_Comm_compare(communicator, X.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for X and the "
                        "input triangulation"));
      ierr =
        MPI_Comm_compare(communicator, F_rhs.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for F_rhs and "
                        "the input triangulation"));
    }
#endif

    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.current_f_data_idx = f_data_idx;

    // Setup X info:
    transaction.native_X_dof_handler = &X_dof_handler;
    transaction.native_X             = &X;
    transaction.overlap_X_vec.reinit(
      get_overlap_dof_handler(X_dof_handler).n_dofs());

    // Setup F info:
    transaction.native_F_dof_handler = &F_dof_handler;
    transaction.F_mapping            = &F_mapping;
    transaction.native_F_rhs         = &F_rhs;
    transaction.overlap_F.reinit(
      get_overlap_dof_handler(F_dof_handler).n_dofs());

    // Setup state:
    transaction.next_state = Transaction<dim, spacedim>::State::Intermediate;
    transaction.operation =
      Transaction<dim, spacedim>::Operation::Interpolation;

    // OK, now start scattering:
    Scatter<double> &X_scatter = get_scatter(X_dof_handler);

    // Since we set up our own communicator in this object we can fearlessly use
    // channels 0 and 1 to guarantee traffic is not accidentally mingled
    int channel = 0;
    X_scatter.global_to_overlap_start(*transaction.native_X,
                                      channel,
                                      transaction.overlap_X_vec);
    ++channel;

    return t_ptr;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation ==
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state ==
            Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    Scatter<double> &X_scatter = get_scatter(*trans.native_X_dof_handler);

    X_scatter.global_to_overlap_finish(*trans.native_X, trans.overlap_X_vec);

    // this is the point at which a base class would normally do computations.

    // After we compute we begin the scatter back to the native partitioning:
    Scatter<double> &F_scatter = get_scatter(*trans.native_F_dof_handler);

    // This object *cannot* get here without the first two scatters finishing so
    // using channel 0 again is fine
    int channel = 0;
    F_scatter.overlap_to_global_start(trans.overlap_F,
                                      VectorOperation::add,
                                      channel,
                                      *trans.native_F_rhs);

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

    Scatter<double> &F_scatter = get_scatter(*trans.native_F_dof_handler);
    F_scatter.overlap_to_global_finish(trans.overlap_F,
                                       VectorOperation::add,
                                       *trans.native_F_rhs);
    trans.next_state = Transaction<dim, spacedim>::State::Done;
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_spread_start(
    const int                                         f_data_idx,
    const LinearAlgebra::distributed::Vector<double> &X,
    const DoFHandler<dim, spacedim> &                 X_dof_handler,
    const Mapping<dim, spacedim> &                    F_mapping,
    const DoFHandler<dim, spacedim> &                 F_dof_handler,
    const LinearAlgebra::distributed::Vector<double> &F)
  {
#ifdef DEBUG
    {
      int result = 0;
      int ierr =
        MPI_Comm_compare(communicator, X.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for X and the "
                        "input triangulation"));
      ierr = MPI_Comm_compare(communicator, F.get_mpi_communicator(), &result);
      AssertThrowMPI(ierr);
      Assert(result == MPI_CONGRUENT,
             ExcMessage("The same communicator should be used for F and the "
                        "input triangulation"));
    }
#endif

    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.current_f_data_idx = f_data_idx;

    // Setup X info:
    transaction.native_X_dof_handler = &X_dof_handler;
    transaction.native_X             = &X;
    transaction.overlap_X_vec.reinit(
      get_overlap_dof_handler(X_dof_handler).n_dofs());

    // Setup F info:
    transaction.native_F_dof_handler = &F_dof_handler;
    transaction.F_mapping            = &F_mapping;
    transaction.native_F             = &F;
    transaction.overlap_F.reinit(
      get_overlap_dof_handler(F_dof_handler).n_dofs());

    // Setup state:
    transaction.next_state = Transaction<dim, spacedim>::State::Intermediate;
    transaction.operation  = Transaction<dim, spacedim>::Operation::Spreading;

    // OK, now start scattering:

    // Since we set up our own communicator in this object we can fearlessly use
    // channels 0 and 1 to guarantee traffic is not accidentally mingled
    int              channel   = 0;
    Scatter<double> &X_scatter = get_scatter(X_dof_handler);
    X_scatter.global_to_overlap_start(*transaction.native_X,
                                      channel,
                                      transaction.overlap_X_vec);
    ++channel;

    Scatter<double> &F_scatter = get_scatter(F_dof_handler);
    F_scatter.global_to_overlap_start(*transaction.native_F,
                                      channel,
                                      transaction.overlap_F);

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

    Scatter<double> &X_scatter = get_scatter(*trans.native_X_dof_handler);
    X_scatter.global_to_overlap_finish(*trans.native_X, trans.overlap_X_vec);

    Scatter<double> &F_scatter = get_scatter(*trans.native_F_dof_handler);
    F_scatter.global_to_overlap_finish(*trans.native_F, trans.overlap_F);

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
            Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Spreading"));
    Assert((trans.next_state == Transaction<dim, spacedim>::State::Finish),
           ExcMessage("Transaction state should be Finish"));

    // since no data is moved there is nothing else to do here

    trans.next_state = Transaction<dim, spacedim>::State::Done;
  }

  // instantiations

  template class InteractionBase<NDIM - 1, NDIM>;
  template class InteractionBase<NDIM, NDIM>;
} // namespace fdl
