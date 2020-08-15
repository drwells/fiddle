#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/shared_tria.h>

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

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
  const types::subdomain_id local_native_subdomain_id = rank;
  parallel::shared::Triangulation<2> tria_1(mpi_comm);
  GridGenerator::hyper_ball(tria_1);
  tria_1.refine_global(2);

  {
    GridOut go;
    std::ofstream out("tria-1.eps");
    go.write_eps(tria_1, out);
  }

  {
    // TODO: this approach doesn't yet support triangulations on multiple
    // levels
    Triangulation<2> tria_2;
    std::vector<CellData<2>> cells;
    SubCellData subcell_data;
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

    std::vector<Triangulation<2>::active_cell_iterator> intersecting_shared_cells;
    for (const auto &cell : tria_1.active_cell_iterators())
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

    tria_2.create_triangulation(tria_1.get_vertices(), cells, SubCellData());
    // now fill in details:
    for (auto &cell : tria_2.active_cell_iterators())
      {
        cell->set_user_index(cell->material_id());
        const auto &shared_cell = intersecting_shared_cells[cell->user_index()];
        cell->set_material_id(shared_cell->material_id());
        cell->set_subdomain_id(shared_cell->subdomain_id());
        cell->set_manifold_id(shared_cell->manifold_id());
      }

    // Get some other things that we need
    std::vector<unsigned int> ib_cell_active_indices;
    std::set<unsigned int> relevant_subdomain_ids;
    for (auto &cell : tria_2.active_cell_iterators())
      {
        auto &native_cell = intersecting_shared_cells[cell->user_index()];
        relevant_subdomain_ids.insert(native_cell->subdomain_id());
        ib_cell_active_indices.push_back(native_cell->active_cell_index());
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

    // We know which processors need our dofs and which processors have our
    // dofs.
    //
    // 1. Define some packed format consisting of active cell index, n_dofs,
    //    and dofs themselves.
    // 2. From the packed format calculate the data length required by other
    //    processors and communicate that to receiving processors.
    // 3. Send the dof data.
    {
      GridOut go;
      std::ofstream out("tria-2-" + std::to_string(rank) + ".svg");
      go.write_svg(tria_2, out);
    }
  }
}
