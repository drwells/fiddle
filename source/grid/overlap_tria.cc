#include <fiddle/grid/overlap_tria.h>

#include <deal.II/grid/tria_description.h>

#include <algorithm>

namespace fdl
{
  using namespace dealii;

  template <int dim, int spacedim>
  OverlapTriangulation<dim, spacedim>::OverlapTriangulation(
    const parallel::shared::Triangulation<dim, spacedim> &shared_tria,
    const IntersectionPredicate<dim, spacedim>           &predicate)
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
    const IntersectionPredicate<dim, spacedim>           &predicate)
  {
    // todo - clear signals, etc. if there is a new shared tria
    native_tria = &shared_tria;

    reinit_overlapping_tria(predicate);
  }



  template <int dim, int spacedim>
  void
  OverlapTriangulation<dim, spacedim>::reinit_overlapping_tria(
    const IntersectionPredicate<dim, spacedim> &predicate)
  {
    native_cells.clear();
    native_cell_ids.clear();
    native_cell_subdomain_ids.clear();
    this->clear();

    const auto native_manifold_ids = native_tria->get_manifold_ids();

    std::vector<CellData<dim>> cells;
    unsigned int               coarsest_level_n = numbers::invalid_unsigned_int;
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
    if (coarsest_level_n != numbers::invalid_unsigned_int)
      {
        for (const auto &cell :
             native_tria->cell_iterators_on_level(coarsest_level_n))
          if (predicate(cell))
            {
              CellData<dim> cell_data(cell->n_vertices());
              cell_data.manifold_id = cell->manifold_id();
              // Temporarily refer to native cells with the material id
              cell_data.material_id = add_native_cell(cell);

              cell_data.vertices.clear();
              for (const auto &index : cell->vertex_indices())
                cell_data.vertices.push_back(cell->vertex_index(index));
              cells.push_back(std::move(cell_data));
            }
      }
    else
      {
        // We don't intersect with any cells, which is a problem - deal.II does
        // not support setting up serial Triangulation objects with zero cells.
        // Add just the first active cell to get around this issue:
        coarsest_level_n = native_tria->n_levels() - 1;
        const auto cell  = native_tria->begin_active(coarsest_level_n);

        CellData<dim> cell_data(0);
        cell_data.vertices.reserve(cell->n_vertices());
        cell_data.manifold_id = cell->manifold_id();
        cell_data.material_id = add_native_cell(cell);
        for (const auto &index : cell->vertex_indices())
          cell_data.vertices.push_back(cell->vertex_index(index));
        cells.push_back(std::move(cell_data));
        // TODO - should we bother setting up boundary data?
      }
    // Set up the coarsest level of the new overlap triangulation:
    this->create_triangulation(native_tria->get_vertices(),
                               cells,
                               SubCellData());
    for (const auto &manifold_id : native_manifold_ids)
      {
        if (manifold_id != numbers::flat_manifold_id)
          this->set_manifold(manifold_id,
                             native_tria->get_manifold(manifold_id));
      }

    auto copy_cell_properties = [&](const auto &native_cell, auto &cell)
    {
      cell->set_material_id(native_cell->material_id());
      cell->set_manifold_id(native_cell->manifold_id());

      // These should be equal but might merely be close after we refine the
      // grid a few times.
      Assert((cell->center() - native_cell->center()).norm() < 1e-12,
             ExcFDLInternalError());
      Assert(cell->n_faces() == native_cell->n_faces(), ExcInternalError());
      for (const auto &face_n : cell->face_indices())
        {
          auto face        = cell->face(face_n);
          auto native_face = native_cell->face(face_n);
          // The actual vertex order might be different due to different
          // orientations: that's checked for consistency below when we set line
          // information
          Assert((face->center() - native_face->center()).norm() < 1e-12,
                 ExcFDLInternalError());
          face->set_manifold_id(native_face->manifold_id());
          if (face->at_boundary())
            {
              if (native_face->at_boundary())
                face->set_boundary_id(native_face->boundary_id());
              else
                face->set_boundary_id(internal_boundary_id);
            }
        }
      // Important: the face orientations may not be the same between the two
      // Triangulations, so looping across these per-face is wrong. The lines
      // have to match up - if not we cannot correctly translate DoF support
      // points
      if (dim > 2)
        for (const auto &line_n : cell->line_indices())
          {
            auto line        = cell->line(line_n);
            auto native_line = native_cell->line(line_n);
            Assert((line->vertex(0) - native_line->vertex(0)).norm() < 1e-12 &&
                     (line->vertex(1) - native_line->vertex(1)).norm() < 1e-12,
                   ExcFDLInternalError());

            line->set_manifold_id(native_line->manifold_id());
            if (line->at_boundary())
              {
                if (native_line->at_boundary())
                  line->set_boundary_id(native_line->boundary_id());
                else
                  line->set_boundary_id(internal_boundary_id);
              }
          }
    };

    for (auto &cell : this->active_cell_iterators())
      {
        // switch the material id for the user index so that native cell
        // lookup works:
        cell->set_user_index(cell->material_id());
        const auto native_cell = get_native_cell(cell);
        copy_cell_properties(native_cell, cell);
      }

    for (unsigned int level_n = 0;
         level_n < native_tria->n_levels() - coarsest_level_n;
         ++level_n)
      {
        // If a native cell is refined then mark the equivalent overlap cell
        // for refinement.
        bool do_refinement = false;
        for (auto &cell : this->cell_iterators_on_level(level_n))
          {
            const auto native_cell = get_native_cell(cell);
            if (predicate(native_cell))
              {
                cell->set_subdomain_id(0);
                if (native_cell->has_children())
                  {
                    do_refinement = true;
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
        if (do_refinement)
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
                        Assert((child->center() - native_child->center())
                                   .norm() < 1e-12,
                               ExcFDLInternalError());
                        child->set_user_index(add_native_cell(native_child));
                        child->set_subdomain_id(0);
                        if (native_child->is_active())
                          copy_cell_properties(native_child, child);
                      }
                  }
              }
          }

        // Check to see if we have run out of levels - i.e., if we didn't refine
        // and we are on the finest level, we can't look for cells to refine on
        // the next level (as it does not exist)
        if (level_n == this->n_levels() - 1)
          break;
      }
  }

  template class OverlapTriangulation<NDIM - 1, NDIM>;
  template class OverlapTriangulation<NDIM, NDIM>;
} // namespace fdl
