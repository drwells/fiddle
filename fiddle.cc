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

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

// todo - add it to deal.II
template <int spacedim>
bool intersects(const BoundingBox<spacedim> &a,
                const BoundingBox<spacedim> &b)
{
    // Since boxes are tensor products of line intervals it suffices to check
    // that the line segments for each coordinate axis overlap.
    for (unsigned int d = 0; d < spacedim; ++d)
    {
        // Line segments can intersect in two ways:
        // 1. They can overlap.
        // 2. One can be inside the other.
        //
        // In the first case we want to see if either end point of the second
        // line segment lies within the first. In the second case we can simply
        // check that one end point of the first line segment lies in the second
        // line segment. Note that we don't need, in the second case, to do two
        // checks since that case is already covered by the first.
        if (!((a.lower_bound(d) <= b.lower_bound(d) && b.lower_bound(d) <= a.upper_bound(d)) ||
              (a.lower_bound(d) <= b.upper_bound(d) && b.upper_bound(d) <= a.upper_bound(d))) &&
            !((b.lower_bound(d) <= a.lower_bound(d) && a.lower_bound(d) <= b.upper_bound(d))))
        {
            return false;
        }
    }

    return true;
}

// convenience class for converting between a local and global set of numbers.
class IndexTranslator
{
  public:
    IndexTranslator(const std::vector<types::global_dof_index> &consecutive_dofs,
                    const std::vector<types::global_dof_index> &nonconsecutive_dofs)
      : nonconsecutive_to_internal_index(nonconsecutive_dofs.size() == 0 ? 0  :
                                         *std::max_element(nonconsecutive_dofs.begin(),
                                                           nonconsecutive_dofs.end()))
    {
      Assert(consecutive_dofs.size() == nonconsecutive_dofs.size(),
             ExcMessage("should have same length"));

      // Set up the mapping from consecutive dofs to nonconsecutive dofs.
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>> temp;
      for (std::size_t i = 0; i < consecutive_dofs.size(); ++i)
        temp.emplace_back(consecutive_dofs[i], nonconsecutive_dofs[i]);
      std::sort(temp.begin(), temp.end(),
                [&](const auto &a, const auto &b)
                {
                  return a.first < b.first;
                });
      for (const auto &pair : temp) consecutive_to_non.push_back(pair.second);

      // Set up the mapping from nonconsecutive to consecutive dofs.
      std::sort(temp.begin(), temp.end(),
                [](const auto &a, const auto &b)
                {
                  return a.second < b.second;
                });
      for (const auto &dof : nonconsecutive_dofs)
        nonconsecutive_to_internal_index.add_index(dof);

      nonconsecutive_to_internal_index.compress();
      for (const auto &index : nonconsecutive_to_internal_index)
      {
        // Find the consecutive dof corresponding to the current nonconsecutive
        // dof.
        const auto it = std::lower_bound(
          temp.begin(), temp.end(), index,
          [](const std::pair<types::global_dof_index, types::global_dof_index> &a,
             const types::global_dof_index &b)
        {
          return a.second < b;
        });
        // push back the entry for faster access in non_to_con:
        internal_to_consecutive.push_back(it->first);
      }
    }

    types::global_dof_index
    con_to_non(const types::global_dof_index a)
    {
      Assert(a < consecutive_to_non.size(), ExcMessage("a too big"));
      return consecutive_to_non[a];
    }


    types::global_dof_index
    non_to_con(const types::global_dof_index b)
    {
      return internal_to_consecutive[nonconsecutive_to_internal_index.index_within_set(b)];
    }


  protected:
    std::vector<types::global_dof_index> consecutive_to_non;

    IndexSet nonconsecutive_to_internal_index;
    std::vector<types::global_dof_index> internal_to_consecutive;
};


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

  {
    // TODO: this approach doesn't yet support triangulations on multiple
    // levels
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

    std::vector<CellData<2>> cells;
    std::vector<Triangulation<2>::active_cell_iterator> intersecting_shared_cells;
    SubCellData subcell_data;
    for (const auto &cell : native_tria.active_cell_iterators())
      {
        if (intersects(cell->bounding_box(), bbox))
          {
            CellData<2> cell_data;
            // abuse the material id to refer to the cells on the shared
            // triangulation. We can fix the actual material values later.
            intersecting_shared_cells.push_back(cell);
            // refer to the last entry
            cell_data.material_id = intersecting_shared_cells.size() - 1;

            // set up vertices:
            cell_data.vertices.clear();
            for (const auto &index : cell->vertex_indices())
              cell_data.vertices.push_back(cell->vertex_index(index));

            cells.push_back(std::move(cell_data));

            // TODO: also populate subcell data to get correct manifold ids
            // and boundary ids

            // TODO: it should suffice to add, if the cell is not on level 0,
            // all family members of the cell to get a real Triangulation
            // working. This will also require some way to duplicate AMR so
            // that we end up with exactly the same part of a Triangulation
          }
      }

    Triangulation<2> ib_tria;
    ib_tria.create_triangulation(native_tria.get_vertices(), cells, SubCellData());
    {
      GridOut go;
      std::ofstream out("tria-2-" + std::to_string(rank) + ".eps");
      go.write_eps(ib_tria, out);
    }

    // now fill in details:
    for (auto &cell : ib_tria.active_cell_iterators())
      {
        cell->set_user_index(cell->material_id());
        const auto &shared_cell = intersecting_shared_cells[cell->user_index()];
        cell->set_material_id(shared_cell->material_id());
        cell->set_subdomain_id(shared_cell->subdomain_id());
        cell->set_manifold_id(shared_cell->manifold_id());
      }

    // Get some other things that we need
    std::set<unsigned int> relevant_subdomain_ids;
    std::map<unsigned int, std::vector<unsigned int>> subdomain_id_to_active_cell_index;
    for (auto &cell : ib_tria.active_cell_iterators())
      {
        auto &native_cell = intersecting_shared_cells[cell->user_index()];
        const auto subdomain_id = native_cell->subdomain_id();
        relevant_subdomain_ids.insert(native_cell->subdomain_id());
        subdomain_id_to_active_cell_index[subdomain_id].push_back
          (native_cell->active_cell_index());
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
        auto &native_cell = intersecting_shared_cells[cell->user_index()];
        native_active_cell_ids_on_ib.push_back(native_cell->active_cell_index());
      }
    std::sort(native_active_cell_ids_on_ib.begin(),
              native_active_cell_ids_on_ib.end());
    std::vector<types::global_dof_index> dofs_on_native;

    // Extract the requested dofs
    {
      std::vector<DoFHandler<2>::active_cell_iterator> cell_iterators;
      for (const auto &cell : native_dof_handler.active_cell_iterators())
        cell_iterators.push_back(cell);
      std::sort(cell_iterators.begin(), cell_iterators.end(),
                [](const auto &a, const auto &b){return a->active_cell_index()
                    < b->active_cell_index();});

      auto active_cell_index_ptr = native_active_cell_ids_on_ib.begin();
      std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
      // there are at least as many cells as requested cells
      for (const auto &cell : cell_iterators)
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
          // include a sentinel
          dofs_on_native.push_back(numbers::invalid_dof_index);

        ++active_cell_index_ptr;
      }
    }

    // todo - when we support multiple processors we will need some merge sort
    // step in here to combine things into a single giant packed array

    // we now have the native dofs on each cell in a packed format: active cell
    // index, dofs, sentinel.

    std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
    {
      std::vector<DoFHandler<2>::active_cell_iterator> active_cells;
      std::vector<types::global_dof_index> ib_dofs;
      std::vector<types::global_dof_index> native_dofs;

      for (const auto &cell : ib_dof_handler.active_cell_iterators())
        active_cells.push_back(cell);

      // sort by active cell indices of the native triangulation:
      std::sort(active_cells.begin(), active_cells.end(),
                [&](const auto &a, const auto &b){return
                    intersecting_shared_cells[a->user_index()]->active_cell_index() <
                      intersecting_shared_cells[b->user_index()]->active_cell_index();});

      auto packed_ptr = dofs_on_native.begin();
      std::vector<types::global_dof_index> native_cell_dofs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> ib_cell_dofs(fe.dofs_per_cell);
      Vector<double> ib_solution(ib_dof_handler.n_dofs());
      for (const auto &cell : active_cells)
      {
        auto &native_cell = intersecting_shared_cells[cell->user_index()];
        Assert(*packed_ptr == native_cell->active_cell_index(), ExcInternalError());
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
        for (unsigned int i = 0; i < ib_cell_dofs.size(); ++i)
        {
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
  }
}
