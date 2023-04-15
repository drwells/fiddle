#include <fiddle/transfer/scatter.h>

#include <deal.II/base/mpi.h>

#include "../tests.h"

// Basic scatter test:
// 1. verify that scattering overlap dofs works correctly
// 2. verify that overlap -> global -> overlap insert is the identity operation

int
main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm   comm    = MPI_COMM_WORLD;
  const auto rank    = dealii::Utilities::MPI::this_mpi_process(comm);
  const auto n_procs = dealii::Utilities::MPI::n_mpi_processes(comm);

  const unsigned int dofs_per_proc = 100;
  const unsigned int n_overlap_dofs_per_proc =
    dofs_per_proc + 10 * (n_procs - 1);
  const auto n_dofs = dofs_per_proc * n_procs;
  IndexSet   local_indices(n_dofs);
  local_indices.add_range(rank * dofs_per_proc, (rank + 1) * dofs_per_proc);
  local_indices.compress();

  // do a simple permutation with modular arithmetic and a coprime step size
  std::vector<types::global_dof_index> permuted_global_dofs;
  types::global_dof_index              index = 0;
  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      permuted_global_dofs.push_back(index % n_dofs);
      index += 41;
    }
  // verify that we really got a permutation:
  {
    std::set<types::global_dof_index> check(permuted_global_dofs.begin(),
                                            permuted_global_dofs.end());
    AssertThrow(check.size() == permuted_global_dofs.size(),
                fdl::ExcFDLInternalError());
  }

  std::vector<types::global_dof_index> overlap_dofs(n_overlap_dofs_per_proc);
  for (unsigned int i = 0; i < n_overlap_dofs_per_proc; ++i)
    overlap_dofs[i] =
      permuted_global_dofs[(n_overlap_dofs_per_proc * rank + i) %
                           permuted_global_dofs.size()];

  // verify that there are no duplicated dofs in overlap_dofs:
  {
    std::set<types::global_dof_index> check(overlap_dofs.begin(),
                                            overlap_dofs.end());
    AssertThrow(check.size() == overlap_dofs.size(),
                fdl::ExcFDLInternalError());
  }

  LinearAlgebra::distributed::Vector<double> global(local_indices, comm);
  for (unsigned int i = 0; i < global.locally_owned_size(); ++i)
    global.local_element(i) = rank * dofs_per_proc + i;
  Vector<double> overlap(n_overlap_dofs_per_proc);

  fdl::Scatter<double> scatter(overlap_dofs, local_indices, comm);
  scatter.global_to_overlap_start(global, 0, overlap);
  scatter.global_to_overlap_finish(global, overlap);

  std::ostringstream out;
  out << "rank = " << rank << '\n';
  bool overlap_equal = true;
  for (unsigned int i = 0; i < n_overlap_dofs_per_proc; ++i)
    overlap_equal = overlap_equal && (overlap_dofs[i] == overlap[i]);
  out << "overlap vector is correct : " << overlap_equal << std::endl;

  LinearAlgebra::distributed::Vector<double> global2(local_indices, comm);
  scatter.overlap_to_global_start(overlap, VectorOperation::insert, 0, global2);
  scatter.overlap_to_global_finish(overlap, VectorOperation::insert, global2);

  bool global_equal = true;
  for (unsigned int i = 0; i < dofs_per_proc; ++i)
    global_equal =
      global_equal && (global2.local_element(i) == global.local_element(i));
  out << "global vectors are equal : " << global_equal << std::endl;

  std::ofstream output;
  if (rank == 0)
    output.open("output");
  print_strings_on_0(out.str(), comm, output);
}
