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
    const Triangulation<dim, spacedim> &native_tria =
      overlap_tria.get_native_triangulation();
    const MPI_Comm mpi_comm = native_tria.get_communicator();
    Assert(&overlap_dof_handler.get_triangulation() == &overlap_tria,
           ExcMessage("The overlap DoFHandler should use the overlap tria"));
    Assert(&native_dof_handler.get_triangulation() == &native_tria,
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
        const types::subdomain_id             requested_rank = pair.first;
        std::vector<types::global_dof_index> &requested_dofs =
          dofs_on_native[requested_rank];
        const std::vector<CellId> &          cell_ids = pair.second;
        std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
        for (const auto &id : cell_ids)
          {
            const auto native_cell = native_tria.create_cell_iterator(id);
            const auto native_dh_cell =
              typename DoFHandler<dim, spacedim>::active_cell_iterator(
                &native_tria,
                native_cell->level(),
                native_cell->index(),
                &native_dof_handler);

            // TODO - we can compress this with 64-bit indices some more - might
            // be worth doing
            const auto binary_id = id.to_binary<dim>();
            for (const auto &v : binary_id)
              requested_dofs.push_back(v);

            native_dh_cell->get_dof_indices(cell_dofs);
            requested_dofs.push_back(cell_dofs.size());
            for (const auto dof : cell_dofs)
              requested_dofs.push_back(dof);

            requested_dofs.push_back(numbers::invalid_dof_index);
          }
      }

    const std::map<types::subdomain_id, std::vector<types::global_dof_index>>
      native_dof_indices =
        Utilities::MPI::some_to_some(mpi_comm, dofs_on_native);

    // we now have the native dofs on each cell in a packed format: active cell
    // index, dofs, sentinel. Read dof data back in the order in which it was
    // originally requested:
    std::map<types::subdomain_id,
             std::vector<types::global_dof_index>::const_iterator>
      packed_ptrs;
    for (const auto &pair : native_dof_indices)
      packed_ptrs[pair.first] = pair.second.cbegin();

    // 4:
    std::vector<types::global_dof_index> overlap_cell_dofs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> native_indices(
      overlap_dof_handler.n_dofs());
    for (const auto &cell : overlap_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            const auto native_rank =
              overlap_tria.get_native_cell_subdomain_id(cell);
            Assert(native_rank != numbers::invalid_subdomain_id,
                   ExcFDLInternalError());
            auto &packed_ptr = packed_ptrs.at(native_rank);
            Assert(packed_ptr < native_dof_indices.at(native_rank).cend(),
                   ExcFDLInternalError());

            CellId::binary_type binary_id;
            for (auto &v : binary_id)
              {
                v = *packed_ptr;
                ++packed_ptr;
              }
#ifdef DEBUG
            {
              const CellId cell_id(binary_id);
              Assert(overlap_tria.get_native_cell_id(cell) == cell_id,
                     ExcFDLInternalError());
            }
#endif

            const auto n_dofs = *packed_ptr;
            ++packed_ptr;
            AssertDimension(n_dofs, overlap_cell_dofs.size());

            // Copy data between orderings.
            cell->get_dof_indices(overlap_cell_dofs);
            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                native_indices[overlap_cell_dofs[i]] = *packed_ptr;
                ++packed_ptr;
              }
            Assert(*packed_ptr == numbers::invalid_dof_index,
                   ExcFDLInternalError());
            ++packed_ptr;
          }
      }

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
