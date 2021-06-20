#include <fiddle/grid/overlap_tria.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim = dim>
  std::vector<types::global_dof_index>
  compute_overlap_to_native_dof_translation(
    const fdl::OverlapTriangulation<dim, spacedim> &overlap_tria,
    const DoFHandler<dim, spacedim> &               overlap_dof_handler,
    const DoFHandler<dim, spacedim> &               native_dof_handler)
  {
    const MPI_Comm mpi_comm =
      overlap_tria.get_native_triangulation().get_communicator();
    Assert(&overlap_dof_handler.get_triangulation() == &overlap_tria,
           ExcMessage("The overlap DoFHandler should use the overlap tria"));
    Assert(&native_dof_handler.get_triangulation() ==
             &overlap_tria.get_native_triangulation(),
           ExcMessage("The native DoFHandler should use the native tria"));
    // Outline of the algorithm:
    //
    // 1. Determine which active cell indices the overlap tria needs.
    //
    // 2. Send (sorted) active cell indices. Use some_to_some for
    //    convenience.
    //
    // We now know who wants which dofs.
    //
    // 3. Pack the requested DoFs and send them back (use some_to_some
    //    again).
    //
    // 4. Loop over active cells to create the mapping between overlap dofs
    //    (purely local) and native dofs (distributed).

    // 1: determine required active cell indices:
    std::map<types::subdomain_id, std::vector<CellId>>
      native_cell_ids_on_overlap;
    for (const auto &cell : overlap_tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto rank = overlap_tria.get_native_cell_subdomain_id(cell);
          // We should never have an active cell which does not have a
          // subdomain: that would mean that the corresponding native cell is
          // not active!
          Assert(rank != numbers::invalid_subdomain_id, ExcFDLInternalError());
          native_cell_ids_on_overlap[rank].push_back(
            overlap_tria.get_native_cell_id(cell));
        }

    // 2: send requested active cell indices:
    const std::map<types::subdomain_id, std::vector<CellId>>
      requested_native_cell_ids =
        Utilities::MPI::some_to_some(mpi_comm, native_cell_ids_on_overlap);

    // 3: pack dofs:
    std::map<types::subdomain_id, std::vector<types::global_dof_index>>
      dofs_on_native;
    const auto &fe = native_dof_handler.get_fe();
    Assert(fe.get_name() == overlap_dof_handler.get_fe().get_name(),
           ExcMessage("dof handlers should use the same FiniteElement"));
    for (const auto &pair : requested_native_cell_ids)
      {
        const types::subdomain_id            requested_rank = pair.first;
        const std::vector<CellId> &          cell_ids       = pair.second;
        std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
        for (const auto &id : cell_ids)
          {
            const auto native_cell =
              overlap_tria.get_native_triangulation().create_cell_iterator(id);
            const auto native_dh_cell =
              typename DoFHandler<dim, spacedim>::active_cell_iterator(
                &overlap_tria.get_native_triangulation(),
                native_cell->level(),
                native_cell->index(),
                &native_dof_handler);
            dofs_on_native[requested_rank].push_back(native_cell->active_cell_index());
            native_dh_cell->get_dof_indices(cell_dofs);
            dofs_on_native[requested_rank].push_back(cell_dofs.size());
            for (const auto dof : cell_dofs)
              dofs_on_native[requested_rank].push_back(dof);

            dofs_on_native[requested_rank].push_back(
              numbers::invalid_dof_index);
          }
      }

    const std::map<types::subdomain_id, std::vector<types::global_dof_index>>
      native_dof_indices =
        Utilities::MPI::some_to_some(mpi_comm, dofs_on_native);

    // we now have the native dofs on each cell in a packed format: active cell
    // index, dofs, sentinel. Read dof data back in the order in which it was
    // originally requested:
    std::map<types::subdomain_id, std::vector<types::global_dof_index>::const_iterator>
      packed_ptrs;
    for (const auto &pair : native_dof_indices)
      packed_ptrs[pair.first] = pair.second.cbegin();

    // 4:
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      overlap_to_native;
    std::vector<types::global_dof_index> native_cell_dofs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> overlap_cell_dofs(fe.dofs_per_cell);
    for (const auto &cell : overlap_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          const auto native_rank = overlap_tria.get_native_cell_subdomain_id(cell);
          Assert(native_rank != numbers::invalid_subdomain_id,
                 ExcFDLInternalError());
          auto &packed_ptr = packed_ptrs.at(native_rank);
          Assert(packed_ptr < native_dof_indices.at(native_rank).cend(),
                 ExcFDLInternalError());
          const auto active_cell_index = *packed_ptr;
          ++packed_ptr;
          Assert(overlap_tria.get_native_cell(cell)->active_cell_index()
                 == active_cell_index,
                 ExcFDLInternalError());

          const auto n_dofs = *packed_ptr;
          ++packed_ptr;

          native_cell_dofs.clear();
          for (unsigned int i = 0; i < n_dofs; ++i)
          {
            native_cell_dofs.push_back(*packed_ptr);
            ++packed_ptr;
          }
          Assert(*packed_ptr == numbers::invalid_dof_index,
                 ExcFDLInternalError());
          ++packed_ptr;

          // Copy data between orderings.
          cell->get_dof_indices(overlap_cell_dofs);
          for (unsigned int i = 0; i < n_dofs; ++i)
            overlap_to_native.emplace_back(overlap_cell_dofs[i],
                                           native_cell_dofs[i]);
        }
      }
    std::sort(overlap_to_native.begin(),
              overlap_to_native.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    overlap_to_native.erase(std::unique(overlap_to_native.begin(),
                                        overlap_to_native.end()),
                            overlap_to_native.end());
    std::vector<types::global_dof_index> native_indices(
      overlap_to_native.size());
    std::transform(overlap_to_native.begin(),
                   overlap_to_native.end(),
                   native_indices.begin(),
                   [](const auto &a) { return a.second; });

    // We finally have the contiguous array native_indices that gives us the
    // native dof for each overlap dof.
    return native_indices;
  }

  template std::vector<types::global_dof_index>
  compute_overlap_to_native_dof_translation(
    const fdl::OverlapTriangulation<NDIM - 1, NDIM> &overlap_tria,
    const DoFHandler<NDIM - 1, NDIM> &               overlap_dof_handler,
    const DoFHandler<NDIM - 1, NDIM> &               native_dof_handler);

  template std::vector<types::global_dof_index>
  compute_overlap_to_native_dof_translation(
    const fdl::OverlapTriangulation<NDIM, NDIM> &overlap_tria,
    const DoFHandler<NDIM, NDIM> &               overlap_dof_handler,
    const DoFHandler<NDIM, NDIM> &               native_dof_handler);
} // namespace fdl
