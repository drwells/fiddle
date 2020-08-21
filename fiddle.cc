#include <deal.II/base/mpi.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fiddle/overlap_triangulation.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
  parallel::shared::Triangulation<2> native_tria(mpi_comm);
  GridGenerator::hyper_ball(native_tria);
  native_tria.refine_global(3);

  {
    GridOut go;
    std::ofstream out("tria-1.eps");
    go.write_eps(native_tria, out);
  }

  BoundingBox<2> bbox;
  switch (rank)
    {
    case 0:
      bbox = BoundingBox<2>(std::make_pair(Point<2>(0.0, 0.0),
                                           Point<2>(1.0, 1.0)));
      break;
    case 1:
      bbox = BoundingBox<2>(std::make_pair(Point<2>(-1.0, 0.0),
                                           Point<2>(0.0, 1.0)));
      break;
    case 2:
      bbox = BoundingBox<2>(std::make_pair(Point<2>(-1.0, -1.0),
                                           Point<2>(0.0, 0.0)));
      break;
    case 3:
      bbox = BoundingBox<2>(std::make_pair(Point<2>(0.0, -1.0),
                                           Point<2>(1.0, 0.0)));
      break;
    default:
      Assert(false, ExcNotImplemented());
    }

  fdl::OverlapTriangulation<2> ib_tria(native_tria, {bbox});

  {
    GridOut go;
    std::ofstream out("tria-2-" + std::to_string(rank) + ".eps");
    go.write_eps(ib_tria, out);
  }

  FE_Q<2> fe(3);
  DoFHandler<2> native_dof_handler(native_tria);
  native_dof_handler.distribute_dofs(fe);
  DoFHandler<2> ib_dof_handler(ib_tria);
  ib_dof_handler.distribute_dofs(fe);

  // go for something even easier - set up the mapping on a single processor first
  Assert(rank == 0, ExcNotImplemented());

  // Possible parallel version:
  //
  // 1. Determine which active cell indices the IB tria needs.
  //
  // 2. Send (sorted) active cell indices. Use some_to_some for convenience.
  //
  // We now know who wants which dofs.
  //
  // 3. Pack the requested DoFs and send them back (use some_to_some again).
  //
  // 4. Set up an IndexTranslator between IB dofs (purely local) and native dofs
  //    (distributed).

  // 1:
  std::map<types::subdomain_id, std::vector<unsigned int>>
    native_active_cell_ids_on_ib;
  for (const auto &cell : ib_tria.active_cell_iterators())
    {
      const auto &native_cell = ib_tria.get_native_cell(cell);
      const auto subdomain_id = native_cell->subdomain_id();
      native_active_cell_ids_on_ib[subdomain_id]
        .push_back(native_cell->active_cell_index());
    }
  for (auto &pair : native_active_cell_ids_on_ib)
    std::sort(pair.second.begin(), pair.second.end());

  // 2:
  const std::map<types::subdomain_id, std::vector<unsigned int>>
    requested_native_active_cell_indices =
    Utilities::MPI::some_to_some(mpi_comm, native_active_cell_ids_on_ib);

  // 3:
  std::map<types::subdomain_id,
           std::vector<types::global_dof_index>> dofs_on_native;
  // TODO: we could make this more efficient by sorting DH iterators by active
  // cell index and then getting them with lower_bound.
  for (const auto &pair : requested_native_active_cell_indices)
  {
    const types::subdomain_id requested_rank = pair.first;
    const std::vector<unsigned int> &active_cell_indices = pair.second;
    auto active_cell_index_ptr = active_cell_indices.cbegin();
    std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
    // Note that we iterate over active cells in order
    for (const auto &cell : native_dof_handler.active_cell_iterators())
    {
      if (cell->active_cell_index() != *active_cell_index_ptr)
        continue;

      Assert(active_cell_index_ptr < active_cell_indices.end(),
             ExcInternalError());
      dofs_on_native[requested_rank].push_back(*active_cell_index_ptr);
      cell->get_dof_indices(cell_dofs);
      dofs_on_native[requested_rank].push_back(cell_dofs.size());
        for (const auto dof : cell_dofs)
          dofs_on_native[requested_rank].push_back(dof);

      dofs_on_native[requested_rank].push_back(numbers::invalid_dof_index);
      ++active_cell_index_ptr;
    }
  }

  const std::map<types::subdomain_id, std::vector<types::global_dof_index>>
    native_dof_indices = Utilities::MPI::some_to_some(mpi_comm, dofs_on_native);

  // generate some data so we can check what happens when we move
  Vector<double> native_solution(native_dof_handler.n_dofs());
  Vector<double> ib_solution(ib_dof_handler.n_dofs());
  VectorTools::interpolate(native_dof_handler,
                           Functions::CosineFunction<2>(),
                           native_solution);

  // we now have the native dofs on each cell in a packed format: active cell
  // index, dofs, sentinel. Make it easier to look up local cells by sorting by
  // global active cell indices:
  std::vector<DoFHandler<2>::active_cell_iterator> ib_dh_cells;
  for (const auto &cell : ib_dof_handler.active_cell_iterators())
    ib_dh_cells.push_back(cell);
  std::sort(ib_dh_cells.begin(), ib_dh_cells.end(),
            [&](const auto& a, const auto& b)
            {
              return ib_tria.get_native_cell(a)->active_cell_index() <
                ib_tria.get_native_cell(b)->active_cell_index();
            });

  // 4:
  for (const auto &pair : native_dof_indices)
    {
      const std::vector<types::global_dof_index> &native_dofs = pair.second;
      auto packed_ptr = native_dofs.cbegin();
      std::vector<types::global_dof_index> native_cell_dofs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> ib_cell_dofs(fe.dofs_per_cell);

      while (packed_ptr < native_dofs.cend())
        {
          const auto active_cell_index = *packed_ptr;
          ++packed_ptr;
          const auto ib_dh_cell =
            std::lower_bound(ib_dh_cells.begin(), ib_dh_cells.end(),
                             active_cell_index,
                             [&](const auto &a, const auto &val)
                             {
                               return ib_tria.get_native_cell(a)->active_cell_index()
                                 < val;
                             });
          Assert(ib_dh_cell != ib_dh_cells.end(), ExcInternalError());
          Assert(ib_tria.get_native_cell(*ib_dh_cell)->active_cell_index() ==
                 active_cell_index,
                 ExcInternalError());
          const auto n_dofs = *packed_ptr;
          ++packed_ptr;

          std::cout << "dofs on native:\n";
          native_cell_dofs.clear();
          for (unsigned int i = 0; i < n_dofs; ++i)
          {
            std::cout << *packed_ptr << '\n';
            native_cell_dofs.push_back(*packed_ptr);
            ++packed_ptr;
          }
          Assert(*packed_ptr == numbers::invalid_dof_index, ExcInternalError());
          ++packed_ptr;

          // We should really set up an IndexTranslator here instead
          (*ib_dh_cell)->get_dof_indices(ib_cell_dofs);
          for (unsigned int i = 0; i < n_dofs; ++i)
            ib_solution[ib_cell_dofs[i]] = native_solution[native_cell_dofs[i]];
        }
    }

  {
    DataOut<2> data_out;
    data_out.attach_dof_handler(native_dof_handler);
    data_out.add_data_vector(native_solution, "solution");
    data_out.build_patches();

    std::ofstream out("native-solution.vtu");
    data_out.write_vtu(out);
  }

  {
    DataOut<2> data_out;
    data_out.attach_dof_handler(ib_dof_handler);
    data_out.add_data_vector(ib_solution, "solution");
    data_out.build_patches();

    std::ofstream out("ib-solution.vtu");
    data_out.write_vtu(out);
  }
}
