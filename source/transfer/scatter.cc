#include <fiddle/transfer/scatter.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  IndexSet
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



  template <typename T>
  Scatter<T>::Scatter(const std::vector<types::global_dof_index> &overlap,
                      const IndexSet &                            local_dofs,
                      const MPI_Comm &                            communicator)
    : overlap_dofs(overlap)
    , scatterer(local_dofs,
                setup_ghost_dofs(overlap_dofs, local_dofs),
                communicator)
  {
    Assert(local_dofs.is_contiguous() == true,
           ExcMessage("The index set specified in local_dofs is not "
                      "contiguous."));
#ifdef DEBUG
    for (const types::global_dof_index index : overlap)
      {
        Assert(index < local_dofs.size(),
               ExcMessage("dofs in overlap should be in the global range "
                          "specified by local_dofs"));
      }
#endif
  }



  template <typename T>
  void
  Scatter<T>::overlap_to_global_start(
    const Vector<T> &                      input,
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

    // This requires some care - when we scatter with insert we assume that the
    // dof is set to the correct value on its owning processor. This is not the
    // case here since local values are not present in the overlap space. In
    // particular, neither ghost data nor owned data is known before we
    // communicate so we can't really check for consistent settings (i.e., there
    // is no correct value set on the owning processor). Hence we do a max
    // operation instead and hope the caller isn't doing anything too weird.
    Assert(operation == VectorOperation::insert ||
             operation == VectorOperation::add,
           ExcNotImplemented());
    if (operation == VectorOperation::add)
      scatterer = 0.0;
    else
      {
        const auto size = scatterer.local_size() +
                          scatterer.get_partitioner()->n_ghost_indices();
        for (std::size_t i = 0; i < size; ++i)
          scatterer.local_element(i) = std::numeric_limits<T>::min();
      }
    // TODO: we can probably do the index translation just once and store it
    // so we could instead use scatterer::local_element(). It might be faster
    // but it will take up more memory.
    for (std::size_t i = 0; i < overlap_dofs.size(); ++i)
      scatterer[overlap_dofs[i]] = input[i];
    // The ghost array is out of sync with the actual values the vector should
    // have - explicitly set it as such so we can compress
    scatterer.set_ghost_state(false);

    const VectorOperation::values actual_op =
      operation == VectorOperation::add ? VectorOperation::add :
                                          VectorOperation::max;
    scatterer.compress_start(channel, actual_op);
  }



  template <typename T>
  void
  Scatter<T>::overlap_to_global_finish(
    const Vector<T> &                      input,
    const VectorOperation::values          operation,
    LinearAlgebra::distributed::Vector<T> &output)
  {
    Assert(input.size() == overlap_dofs.size(),
           ExcMessage("Input vector should be indexed by overlap dofs"));
    Assert(output.locally_owned_elements() ==
             scatterer.locally_owned_elements(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    const VectorOperation::values actual_op =
      operation == VectorOperation::add ? VectorOperation::add :
                                          VectorOperation::max;
    scatterer.compress_finish(actual_op);

    for (std::size_t i = 0; i < scatterer.local_size(); ++i)
      output.local_element(i) = scatterer.local_element(i);
  }



  template <typename T>
  void
  Scatter<T>::global_to_overlap_start(
    const LinearAlgebra::distributed::Vector<T> &input,
    const unsigned int                           channel,
    Vector<T> &                                  output)
  {
    Assert(output.size() == overlap_dofs.size(),
           ExcMessage("output vector should be indexed by overlap dofs"));
    Assert(input.locally_owned_elements() == scatterer.locally_owned_elements(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    scatterer.zero_out_ghosts();
    for (std::size_t i = 0; i < scatterer.local_size(); ++i)
      scatterer.local_element(i) = input.local_element(i);

    scatterer.update_ghost_values_start(channel);
  }



  template <typename T>
  void
  Scatter<T>::global_to_overlap_finish(
    const LinearAlgebra::distributed::Vector<T> &input,
    Vector<T> &                                  output)
  {
    Assert(output.size() == overlap_dofs.size(),
           ExcMessage("output vector should be indexed by overlap dofs"));
    Assert(input.locally_owned_elements() == scatterer.locally_owned_elements(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    scatterer.update_ghost_values_finish();

    for (std::size_t i = 0; i < output.size(); ++i)
      output[i] = scatterer[overlap_dofs[i]];
  }

  template class Scatter<float>;
  template class Scatter<double>;
} // namespace fdl
