#include <fiddle/base/exceptions.h>

#include <fiddle/transfer/scatter.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  IndexSet
  setup_ghost_dofs(const std::vector<types::global_dof_index> &overlap_dofs,
                   const IndexSet                             &local_dofs)
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
  Scatter<T>::Scatter()
    : partitioner(std::make_shared<Utilities::MPI::Partitioner>())
    , n_overlap_dofs(0)
  {}

  template <typename T>
  Scatter<T>::Scatter(const std::vector<types::global_dof_index> &overlap_dofs,
                      const IndexSet                             &local_dofs,
                      const MPI_Comm                             &communicator)
    : partitioner(std::make_shared<Utilities::MPI::Partitioner>(
        local_dofs,
        setup_ghost_dofs(overlap_dofs, local_dofs),
        communicator))
    , n_overlap_dofs(overlap_dofs.size())
    , ghost_buffer(partitioner->n_ghost_indices())
    , import_buffer(partitioner->n_import_indices())
  {
    Assert(local_dofs.is_contiguous() == true,
           ExcMessage("The index set specified in local_dofs is not "
                      "contiguous."));
#ifdef DEBUG
    for (const types::global_dof_index index : overlap_dofs)
      {
        Assert(index < local_dofs.size(),
               ExcMessage("dofs in overlap should be in the global range "
                          "specified by local_dofs"));
      }
#endif
    std::vector<std::pair<unsigned int, unsigned int>> overlap_pairs;
    for (unsigned int i = 0; i < overlap_dofs.size(); ++i)
      if (!partitioner->in_local_range(overlap_dofs[i]))
        {
          overlap_pairs.emplace_back(
            partitioner->ghost_indices().index_within_set(overlap_dofs[i]), i);
        }
    Assert(overlap_pairs.size() == partitioner->ghost_indices().n_elements(),
           ExcFDLInternalError());
    std::sort(overlap_pairs.begin(),
              overlap_pairs.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (unsigned int i = 0; i < overlap_pairs.size(); ++i)
      {
        Assert(overlap_pairs[i].first == i, ExcFDLInternalError());
        overlap_ghost_indices.push_back(overlap_pairs[i].second);
      }

    for (std::size_t i = 0; i < overlap_dofs.size(); ++i)
      if (partitioner->in_local_range(overlap_dofs[i]))
        overlap_local_indices.emplace_back(
          i, partitioner->global_to_local(overlap_dofs[i]));
  }



  template <typename T>
  void
  Scatter<T>::overlap_to_global_start(
    const Vector<T>                       &input,
    const VectorOperation::values          operation,
    const unsigned int                     channel,
    LinearAlgebra::distributed::Vector<T> &output)
  {
    Assert(input.size() == n_overlap_dofs,
           ExcMessage("Input vector should be indexed by overlap dofs"));
    Assert(output.locally_owned_size() == partitioner->locally_owned_size(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    // This requires some care - when we scatter with insert we assume that the
    // dof is set to the correct value on its owning processor. This is not the
    // case here since local values are not present in the overlap space. In
    // particular, neither ghost data nor owned data is known before we
    // communicate so we can't really check for consistent settings (i.e., there
    // is no correct value set on the owning processor). Hence we do a max
    // operation instead and hope the caller isn't doing anything too weird.
    output.set_ghost_state(false);
    if (operation == VectorOperation::add)
      output = 0.0;
    else if (operation == VectorOperation::insert ||
             operation == VectorOperation::max)
      {
        const auto size = output.locally_owned_size() +
                          output.get_partitioner()->n_ghost_indices();
        std::fill(output.get_values(),
                  output.get_values() + size,
                  std::numeric_limits<T>::lowest());
      }
    else
      {
        Assert(false, ExcFDLNotImplemented());
      }

    for (unsigned int i = 0; i < overlap_ghost_indices.size(); ++i)
      ghost_buffer[i] = input[overlap_ghost_indices[i]];

    for (const auto &pair : overlap_local_indices)
      output.local_element(pair.second) = input[pair.first];

    const VectorOperation::values actual_op =
      operation == VectorOperation::insert ? VectorOperation::max : operation;

    partitioner->import_from_ghosted_array_start(
      actual_op,
      channel,
      ArrayView<T>(ghost_buffer.data(), ghost_buffer.size()),
      ArrayView<T>(import_buffer.data(), import_buffer.size()),
      requests);
  }



  template <typename T>
  void
  Scatter<T>::overlap_to_global_finish(
    const Vector<T>                       &input,
    const VectorOperation::values          operation,
    LinearAlgebra::distributed::Vector<T> &output)
  {
    (void)input;
    Assert(input.size() == n_overlap_dofs,
           ExcMessage("Input vector should be indexed by overlap dofs"));
    Assert(output.locally_owned_size() == partitioner->locally_owned_size(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    const VectorOperation::values actual_op =
      operation == VectorOperation::insert ? VectorOperation::max : operation;

    partitioner->import_from_ghosted_array_finish<T>(
      actual_op,
      ArrayView<const T>(import_buffer.data(), import_buffer.size()),
      ArrayView<T>(output.get_values(), output.locally_owned_size()),
      ArrayView<T>(ghost_buffer.data(), ghost_buffer.size()),
      requests);
  }



  template <typename T>
  void
  Scatter<T>::global_to_overlap_start(
    const LinearAlgebra::distributed::Vector<T> &input,
    const unsigned int                           channel,
    Vector<T>                                   &output)
  {
    (void)output;
    Assert(output.size() == n_overlap_dofs,
           ExcMessage("output vector should be indexed by overlap dofs"));
    Assert(input.locally_owned_size() == partitioner->locally_owned_size(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    partitioner->export_to_ghosted_array_start<T>(
      channel,
      ArrayView<const T>(input.get_values(), input.locally_owned_size()),
      ArrayView<T>(import_buffer.data(), import_buffer.size()),
      ArrayView<T>(ghost_buffer.data(), ghost_buffer.size()),
      requests);
  }



  template <typename T>
  void
  Scatter<T>::global_to_overlap_finish(
    const LinearAlgebra::distributed::Vector<T> &input,
    Vector<T>                                   &output)
  {
    Assert(output.size() == n_overlap_dofs,
           ExcMessage("output vector should be indexed by overlap dofs"));
    Assert(input.locally_owned_size() == partitioner->locally_owned_size(),
           ExcMessage("The output vector should have the same number of dofs "
                      "as were provided to the constructor in local"));

    partitioner->export_to_ghosted_array_finish(
      ArrayView<T>(ghost_buffer.data(), ghost_buffer.size()), requests);

    for (unsigned int i = 0; i < overlap_ghost_indices.size(); ++i)
      output[overlap_ghost_indices[i]] = ghost_buffer[i];

    for (const auto &pair : overlap_local_indices)
      output[pair.first] = input.local_element(pair.second);
  }

  template <typename T>
  std::vector<MPI_Request>
  Scatter<T>::delegate_outstanding_requests()
  {
    auto copy = requests;
    std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
    return copy;
  }

  template class Scatter<float>;
  template class Scatter<double>;
} // namespace fdl
