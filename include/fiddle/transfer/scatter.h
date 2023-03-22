#ifndef included_fiddle_transfer_scatter_h
#define included_fiddle_transfer_scatter_h

#include <fiddle/base/config.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  /**
   * LA::d::V-based replacement for PETSc's VecScatter. Moves data from overlap
   * to distributed views.
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
    Scatter() = default;

    /**
     * Move constructor.
     */
    Scatter(Scatter<T> &&) = default;

    /**
     * Move assignment.
     */
    Scatter<T> &
    operator=(Scatter<T> &&) = default;

    /**
     * Constructor.
     */
    Scatter(const std::vector<types::global_dof_index> &overlap,
            const IndexSet                             &local,
            const MPI_Comm                             &communicator);

    /**
     * Scatter a sequential vector indexed by the specified overlap dofs into
     * the
     * parallel distributed vector @p output. Since multiple values may be set for
     * the same dof in the overlap vector a VectorOperation is required to
     * combine
     * values. Ghost values of @p output are not set.
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

  protected:
    std::vector<types::global_dof_index> overlap_dofs;

    LinearAlgebra::distributed::Vector<T> scatterer;
  };
} // namespace fdl
#endif
