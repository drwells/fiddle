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
  const types::subdomain_id local_native_subdomain_id = rank;
  parallel::shared::Triangulation<2> native_tria(mpi_comm);
  GridGenerator::hyper_ball(native_tria);
  native_tria.refine_global(2);

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

#if 0
  // Get some other things that we need
  std::set<unsigned int> relevant_subdomain_ids;
  for (auto &cell : ib_tria.active_cell_iterators())
    {
      auto &native_cell = intersecting_shared_cells[cell->user_index()];
      relevant_subdomain_ids.insert(native_cell->subdomain_id());
    }

  // OK, now we know which cells we need to get dof data from
  //
  // Lets do this in two steps:
  // 1. Figure out what processors we need data from (done in the loop above)
  // 2. Figure out what processors need data from us (done by
  //    compute_point_to_point_communication_pattern)
  relevant_subdomain_ids.erase(local_native_subdomain_id);
  const std::vector<unsigned int> native_procs_used_on_local_ib
    (relevant_subdomain_ids.begin(), relevant_subdomain_ids.end());
  const std::vector<unsigned int> ib_procs_needing_local_native =
    Utilities::MPI::compute_point_to_point_communication_pattern
    (mpi_comm, native_procs_used_on_local_ib);
#endif

  FE_Q<2> fe(3);
  DoFHandler<2> native_dof_handler(native_tria);
  native_dof_handler.distribute_dofs(fe);
  DoFHandler<2> ib_dof_handler(ib_tria);
  ib_dof_handler.distribute_dofs(fe);

  // generate some data so we can check what happens when we move
  Vector<double> native_solution(native_dof_handler.n_dofs());
  VectorTools::interpolate(native_dof_handler,
                           Functions::CosineFunction<2>(),
                           native_solution);

  // go for something even easier - set up the mapping on a single processor first
  Assert(rank == 0, ExcNotImplemented());

  std::vector<unsigned int> native_active_cell_ids_on_ib;
  for (const auto &cell : ib_tria.active_cell_iterators())
    {
      const auto &native_cell = ib_tria.get_native_cell(cell);
      native_active_cell_ids_on_ib.push_back(native_cell->active_cell_index());
    }
  std::sort(native_active_cell_ids_on_ib.begin(),
            native_active_cell_ids_on_ib.end());
  std::vector<types::global_dof_index> dofs_on_native;

  // Extract the requested dofs
  {
    auto active_cell_index_ptr = native_active_cell_ids_on_ib.begin();
    std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
    // Note that we iterate over active cells in order
    for (const auto &cell : native_dof_handler.active_cell_iterators())
    {
      if (cell->active_cell_index() != *active_cell_index_ptr)
        continue;

      Assert(active_cell_index_ptr < native_active_cell_ids_on_ib.end(),
             ExcInternalError());
      dofs_on_native.push_back(*active_cell_index_ptr);
      cell->get_dof_indices(cell_dofs);
      dofs_on_native.push_back(cell_dofs.size());
        for (const auto dof : cell_dofs)
          dofs_on_native.push_back(dof);

      dofs_on_native.push_back(numbers::invalid_dof_index);
      ++active_cell_index_ptr;
    }
  }

  // todo - when we support multiple processors we will need some merge sort
  // step in here to combine things into a single giant packed array

  // we now have the native dofs on each cell in a packed format: active cell
  // index, dofs, sentinel.
  {
    auto packed_ptr = dofs_on_native.begin();
    std::vector<types::global_dof_index> native_cell_dofs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> ib_cell_dofs(fe.dofs_per_cell);
    Vector<double> ib_solution(ib_dof_handler.n_dofs());
    for (const auto &cell : ib_dof_handler.active_cell_iterators())
    {
      const auto &native_cell = ib_tria.get_native_cell(cell);
      Assert(*packed_ptr == native_cell->active_cell_index(),
             ExcInternalError());
      ++packed_ptr;
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

      cell->get_dof_indices(ib_cell_dofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        ib_solution[ib_cell_dofs[i]] = native_solution[native_cell_dofs[i]];
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
}
