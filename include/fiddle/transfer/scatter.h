#ifndef included_fiddle_transfer_scatter_h
#define included_fiddle_transfer_scatter_h

#include <fiddle/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  /**
   * dealii::MPI::Partitioner-based replacement for PETSc's VecScatter. Moves
   * data back-and-forth from the standard 'global' partitioning to the overlap
   * partitioning used for IB calculations.
   *
   * @note instantiations are only available for float and double.
   *
   * @warning like VecScatter, fdl::Scatter objects can only handle one scatter
   * at a time: i.e., after each start the corresponding finish function must be
   * called.
   *
   * @todo Add a constructor taking a dealii::MPI::Partitioner object to share
   * communication data between instances.
   */
  template <typename T>
  class Scatter
  {
  public:
    /**
     * Default constructor. Sets up an empty object over MPI_COMM_SELF.
     */
    Scatter();

    /**
     * Move constructor.
     */
    Scatter(Scatter<T> &&);

    /**
     * Move assignment.
     */
    Scatter<T> &
    operator=(Scatter<T> &&);

    /**
     * Constructor.
     */
    Scatter(const std::vector<types::global_dof_index> &overlap_dofs,
            const IndexSet                             &local,
            const MPI_Comm                             &communicator);

    /**
     * Scatter a sequential vector indexed by the specified overlap dofs into
     * the parallel distributed vector @p output. Since multiple values may be
     * set for the same dof in the overlap vector a VectorOperation is required
     * to combine values. Ghost values of @p output are not set.
     */
    void
    overlap_to_global_start(const Vector<T>                       &input,
                            const VectorOperation::values          operation,
                            const unsigned int                     channel,
                            LinearAlgebra::distributed::Vector<T> &output);

    /**
     * Finish the overlap to global scatter. No ghost values are updated.
     */
    void
    overlap_to_global_finish(const Vector<T>                       &input,
                             const VectorOperation::values          operation,
                             LinearAlgebra::distributed::Vector<T> &output);

    /**
     * Start the scatter from a parallel vector whose data layout matches the
     * index set local provided to the constructor to the overlap vector whose
     * data layout matches the overlap indices provided to the constructor. No
     * ghost data is read from @p input (instead, a temporary array does a ghost
     * update).
     */
    void
    global_to_overlap_start(const LinearAlgebra::distributed::Vector<T> &input,
                            const unsigned int channel,
                            Vector<T>         &output);

    /**
     * Finish the global to overlap scatter.
     */
    void
    global_to_overlap_finish(const LinearAlgebra::distributed::Vector<T> &input,
                             Vector<T> &output);

    /**
     * Delegate responsibility for completing all outstanding MPI requests to
     * some other object (i.e., someone else will call MPI_Waitall() or an
     * equivalent function). The corresponding requests owned by this object
     * will be set to MPI_REQUEST_NULL (i.e., completed requests).
     *
     * When doing multiple concurrent global to overlap scatters, the sum of the
     * scatters is load balanced but individual scatters are not. Hence it is
     * more efficient to wait for all scatters simultaneously than individually.
     */
    std::vector<MPI_Request>
    delegate_outstanding_requests();

  protected:
    std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

    /**
     * Number of DoFs in the overlap partitioning.
     */
    std::size_t n_overlap_dofs;

    /**
     * Indices of overlap_dofs which correspond to entries in the ghost buffer.
     * These are sorted so that the first entry is an index into an
     * overlap-partitioned vector corresponding to the in the first entry of the
     * ghost buffer (i.e., they are contiguous in the ghost buffer and not in
     * the overlap-partitioned vector).
     */
    std::vector<unsigned int> overlap_ghost_indices;

    /**
     * Indices of overlap_dofs which correspond to local (i.e, computed with
     * Partitioner::global_to_local()) locally-owned (i.e., not in the ghost
     * region) indices of the global vector. The first entry in the pair is an
     * index into an overlap-partitioned vector and the second is the local
     * index of a global vector.
     */
    std::vector<std::pair<unsigned int, unsigned int>> overlap_local_indices;

    AlignedVector<T>         ghost_buffer;
    AlignedVector<T>         import_buffer;
    std::vector<MPI_Request> requests;
  };


  // --------------------------- inline functions --------------------------- //


  template <typename T>
  inline Scatter<T>::Scatter(Scatter<T> &&t)
  {
    partitioner.swap(t.partitioner);
    std::swap(n_overlap_dofs, t.n_overlap_dofs);
    overlap_ghost_indices.swap(t.overlap_ghost_indices);
    overlap_local_indices.swap(t.overlap_local_indices);
    ghost_buffer.swap(t.ghost_buffer);
    import_buffer.swap(t.import_buffer);
    requests.swap(t.requests);
  }

  template <typename T>
  inline Scatter<T> &
  Scatter<T>::operator=(Scatter<T> &&t)
  {
    partitioner.swap(t.partitioner);
    std::swap(n_overlap_dofs, t.n_overlap_dofs);
    overlap_ghost_indices.swap(t.overlap_ghost_indices);
    overlap_local_indices.swap(t.overlap_local_indices);
    ghost_buffer.swap(t.ghost_buffer);
    import_buffer.swap(t.import_buffer);
    requests.swap(t.requests);
    return *this;
  }
} // namespace fdl
#endif
