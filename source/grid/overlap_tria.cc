#include <fiddle/grid/overlap_tria.h>

#include <deal.II/grid/tria_description.h>

#include <algorithm>

namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim>
  OverlapTriangulation<dim, spacedim>::OverlapTriangulation(
    const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
    const IntersectionPredicate<dim, spacedim> &          predicate)
  {
    reinit(shared_tria, predicate);
  }



  template <int dim, int spacedim>
  types::subdomain_id
  OverlapTriangulation<dim, spacedim>::locally_owned_subdomain() const
  {
    return 0;
  }



  template <int dim, int spacedim>
  void
  OverlapTriangulation<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
    const IntersectionPredicate<dim, spacedim> &          predicate)
  {
    // todo - clear signals, etc. if there is a new shared tria
    native_tria = &shared_tria;

    reinit_overlapping_tria(predicate);

    // Also set up some cached information:
    cell_iterators_in_active_native_order.clear();
    for (const auto &cell : this->active_cell_iterators())
      if (cell->subdomain_id() != numbers::artificial_subdomain_id)
        cell_iterators_in_active_native_order.push_back(cell);
    std::sort(cell_iterators_in_active_native_order.begin(),
              cell_iterators_in_active_native_order.end(),
              [&](const auto &a, const auto &b) {
                return this->get_native_cell(a)->active_cell_index() <
                       this->get_native_cell(b)->active_cell_index();
              });
  }



  template <int dim, int spacedim>
  void
  OverlapTriangulation<dim, spacedim>::reinit_overlapping_tria(
    const IntersectionPredicate<dim, spacedim> &predicate)
  {
    native_cells.clear();
    this->clear();

    std::vector<CellData<dim>> cells;
    SubCellData                subcell_data;

    unsigned int coarsest_level_n = numbers::invalid_unsigned_int;
    for (unsigned int level_n = 0; level_n < native_tria->n_levels(); ++level_n)
      {
        // We only need to start looking for intersections if the level
        // contains an active cell intersecting the patches.
        for (const auto &cell :
             native_tria->active_cell_iterators_on_level(level_n))
          {
            if (predicate(cell))
              {
                coarsest_level_n = level_n;
                // we need to break out of two loops so jump
                goto found_coarsest_level;
              }
          }
      }

  found_coarsest_level:
    Assert(coarsest_level_n != numbers::invalid_unsigned_int,
           ExcInternalError());
    for (const auto &cell :
         native_tria->cell_iterators_on_level(coarsest_level_n))
      {
        if (predicate(cell))
          {
            CellData<dim> cell_data;
            // Temporarily refer to native cells with the material id
            cell_data.material_id = add_native_cell(cell);

            cell_data.vertices.clear();
            for (const auto &index : cell->vertex_indices())
              cell_data.vertices.push_back(cell->vertex_index(index));
            cells.push_back(std::move(cell_data));

            // set up subcell data:
            auto extract_subcell = [](const auto &iter, auto &cell_data) {
              cell_data.vertices.clear();
              for (const auto &index : iter->vertex_indices())
                cell_data.vertices.push_back(iter->vertex_index(index));
              cell_data.manifold_id = iter->manifold_id();
              cell_data.boundary_id = iter->boundary_id();
            };

            if (dim == 2)
              {
                for (const auto face : cell->face_iterators())
                  {
                    CellData<1> boundary_line;
                    extract_subcell(face, boundary_line);
                    subcell_data.boundary_lines.push_back(
                      std::move(boundary_line));
                  }
              }
            else if (dim == 3)
              {
                for (const auto face : cell->face_iterators())
                  {
                    CellData<2> boundary_quad;
                    extract_subcell(face, boundary_quad);
                    subcell_data.boundary_quads.push_back(
                      std::move(boundary_quad));

                    for (unsigned int line_n = 0; line_n < face->n_lines();
                         ++line_n)
                      {
                        CellData<1> boundary_line;
                        extract_subcell(face->line(line_n), boundary_line);
                        subcell_data.boundary_lines.push_back(
                          std::move(boundary_line));
                      }
                  }
              }
          }
      }

    // Set up the coarsest level of the new overlap triangulation:
    this->create_triangulation(native_tria->get_vertices(),
                               cells,
                               subcell_data);
    for (auto &cell : this->active_cell_iterators())
      {
        // switch the material id for the user index so that native cell
        // lookup works:
        cell->set_user_index(cell->material_id());
        const auto native_cell = get_native_cell(cell);
        cell->set_material_id(native_cell->material_id());
      }
    for (const auto manifold_id : native_tria->get_manifold_ids())
      {
        if (manifold_id != numbers::flat_manifold_id)
          this->set_manifold(manifold_id,
                             native_tria->get_manifold(manifold_id));
      }

    for (unsigned int level_n = 0;
         level_n < native_tria->n_levels() - coarsest_level_n;
         ++level_n)
      {
        // If a native cell is refined then mark the equivalent overlap cell
        // for refinement.
        bool refined = false;
        for (auto &cell : this->cell_iterators_on_level(level_n))
          {
            if (predicate(cell))
              {
                const auto native_cell = get_native_cell(cell);
                cell->set_subdomain_id(0);
                if (native_cell->has_children())
                  {
                    refined = true;
                    Assert(native_cell->refinement_case() ==
                             RefinementCase<dim>::isotropic_refinement,
                           ExcNotImplemented());
                    cell->set_refine_flag();
                  }
              }
            else
              {
                cell->set_subdomain_id(numbers::artificial_subdomain_id);
              }
          }
        if (refined)
          {
            this->execute_coarsening_and_refinement();
            // Copy essential properties to the new cells on level_n + 1 and
            // continue setting up native cells for the new
            // cells.
            for (auto &cell : this->cell_iterators_on_level(level_n))
              {
                if (cell->has_children())
                  {
                    const auto native_cell = get_native_cell(cell);
                    const auto n_children  = cell->n_children();
                    for (unsigned int child_n = 0; child_n < n_children;
                         ++child_n)
                      {
                        auto       child        = cell->child(child_n);
                        const auto native_child = native_cell->child(child_n);
                        // These should be equal but are not after we refine
                        // the grid a few times
                        Assert((child->barycenter() -
                                native_child->barycenter())
                                   .norm() < 1e-12,
                               ExcInternalError());
                        child->set_user_index(add_native_cell(native_child));
                        child->set_subdomain_id(0);
                        if (native_child->is_active())
                          {
                            child->set_material_id(native_child->material_id());
                            child->set_manifold_id(native_child->manifold_id());
                          }
                      }
                  }
              }
          }
      }
  }

  template class OverlapTriangulation<2, 2>;

  template class OverlapTriangulation<2, 3>;

  template class OverlapTriangulation<3, 3>;
} // namespace fdl