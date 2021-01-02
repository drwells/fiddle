#ifndef included_fiddle_scatter_h
#define included_fiddle_scatter_h

#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  /**
   * LA::d::V-based replacement for VecScatter. Moves data from overlap to
   * distributed views.
   */
  template <typename T>
  class Scatter
  {
  public:
    Scatter(const std::vector<types::global_dof_index> &overlap,
            const IndexSet &                            local,
            const MPI_Comm &                            communicator)
      : overlap_dofs(overlap)
      , local_dofs(local)
      , comm(communicator)
      , ghost_dofs(setup_ghost_dofs(overlap_dofs, local_dofs))
      , scatterer(local_dofs, ghost_dofs, communicator)
    {
      Assert(local_dofs.is_contiguous() == true,
             ExcMessage("The index set specified in local_dofs is not "
                        "contiguous."));
#ifdef DEBUG
      for (const types::global_dof_index index : overlap)
        {
          Assert(index < local.size(),
                 ExcMessage("dofs in overlap should be in the global range "
                            "specified by local"));
        }
#endif
    }

    /**
     * Scatter a sequential vector indexed by the specified overlap dofs into
     * the
     * parallel distributed vector @p output. Since multiple values may be set for
     * the same dof in the overlap vector a VectorOperation is required to
     * combine
     * values. Ghost values of @p output are not set.
     */
    void
    overlap_to_global_start(const Vector<T> &                      input,
                            const VectorOperation::values          operation,
                            const unsigned int                     channel,
                            LinearAlgebra::distributed::Vector<T> &output)
    {
      Assert(input.size() == overlap_dofs.size(),
             ExcMessage("Input vector should be indexed by overlap dofs"));
      Assert(output.locally_owned_elements() ==
               scatterer.locally_owned_elements(),
             ExcMessage("The output vector should have the same number of dofs "
                        "as were provided to the constructor in local"));

      Assert(operation == VectorOperation::insert ||
               operation == VectorOperation::add,
             ExcNotImplemented());
      scatterer = 0.0;
      // TODO: we can probably do the index translation just once and store it
      // so we could instead use scatterer::local_element(). It might be faster
      // but it will take up more memory.
      for (std::size_t i = 0; i < overlap_dofs.size(); ++i)
        scatterer[overlap_dofs[i]] = input[i];

      scatterer.compress_start(channel, operation);

      // don't copy any ghost data - ghost regions are not the same anyway
      for (std::size_t i = 0; i < scatterer.local_size(); ++i)
        output.local_element(i) = scatterer.local_element(i);
    }

    /**
     * Finish the overlap to global scatter. No ghost values are updated.
     */
    void
    overlap_to_global_finish(const Vector<T> &                      input,
                             const VectorOperation                  operation,
                             LinearAlgebra::distributed::Vector<T> &output)
    {
      Assert(input.size() == overlap_dofs.size(),
             ExcMessage("Input vector should be indexed by overlap dofs"));
      Assert(output.locally_owned_elements() ==
               scatterer.locally_owned_elements(),
             ExcMessage("The output vector should have the same number of dofs "
                        "as were provided to the constructor in local"));

      scatterer.compress_finish(operation);

      for (std::size_t i = 0; i < scatterer.local_size(); ++i)
        output.local_element(i) = scatterer.local_element(i);
    }


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
                            Vector<T> &        output)
    {
      Assert(output.size() == overlap_dofs.size(),
             ExcMessage("output vector should be indexed by overlap dofs"));
      Assert(input.locally_owned_elements() ==
               scatterer.locally_owned_elements(),
             ExcMessage("The output vector should have the same number of dofs "
                        "as were provided to the constructor in local"));

      scatterer.zero_out_ghosts();
      for (std::size_t i = 0; i < scatterer.local_size(); ++i)
        scatterer.local_element(i) = input.local_element(i);

      scatterer.update_ghost_values_start(channel);
    }

    /**
     * Finish the global to overlap scatter.
     */
    void
    global_to_overlap_finish(const LinearAlgebra::distributed::Vector<T> &input,
                             Vector<T> &output)
    {
      Assert(output.size() == overlap_dofs.size(),
             ExcMessage("output vector should be indexed by overlap dofs"));
      Assert(input.locally_owned_elements() ==
               scatterer.locally_owned_elements(),
             ExcMessage("The output vector should have the same number of dofs "
                        "as were provided to the constructor in local"));

      scatterer.update_ghost_values_finish();

      for (std::size_t i = 0; i < output.size(); ++i)
        output[i] = scatterer[overlap_dofs[i]];
    }

  protected:
    std::vector<types::global_dof_index> overlap_dofs;

    IndexSet local_dofs;

    MPI_Comm comm;

    IndexSet ghost_dofs;

    // TODO: we might want to have more than one scatter in-flight at any given
    // time. Hence we should keep an array of these exemplar vectors that are
    // actually used for scattering
    LinearAlgebra::distributed::Vector<T> scatterer;

    static IndexSet
    setup_ghost_dofs(const std::vector<types::global_dof_index> &overlap_dofs,
                     const IndexSet &                            local_dofs)
    {
      IndexSet ghost_dofs(local_dofs.size());
      // overlap dofs are usually not locally owned, so most of them are
      // ghosts:
      ghost_dofs.add_indices(overlap_dofs.begin(), overlap_dofs.end());
      // remove any overlap dofs that are actually owned by this processor:
      ghost_dofs.subtract_set(local_dofs);
      ghost_dofs.compress();
      return ghost_dofs;
    }
  };
} // namespace fdl
#endif
