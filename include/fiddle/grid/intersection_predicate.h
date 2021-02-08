#ifndef included_fiddle_intersection_predicate_h
#define included_fiddle_intersection_predicate_h

#include <fiddle/grid/box_utilities.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <fiddle/base/exceptions.h>

#include <mpi.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Class which can determine whether or not a given cell intersects some
   * geometric object.
   *
   * At the present time, since fiddle only works with
   * parallel::shared::Triangulation, this class assumes that it can compute an
   * answer for <emph>any</emph> cell in the Triangulation, and not just locally
   * owned cells.
   */
  template <int dim, int spacedim = dim>
  class IntersectionPredicate
  {
  public:
    /**
     * See if a given cell intersects whatever geometric object this object
     * refers to.
     */
    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const = 0;

    virtual ~IntersectionPredicate() = default;
  };

  /**
   * Intersection predicate that determines intersections based on the locations
   * of cells in the Triangulation and nothing else.
   */
  template <int dim, int spacedim = dim>
  class TriaIntersectionPredicate : public IntersectionPredicate<dim, spacedim>
  {
  public:
    TriaIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &bboxes)
      : patch_boxes(bboxes)
    {}

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      const auto cell_bbox = cell->bounding_box();
      for (const auto &bbox : patch_boxes)
        if (intersects(cell_bbox, bbox))
          return true;
      return false;
    }

    const std::vector<BoundingBox<spacedim>> patch_boxes;
  };

  /**
   * Intersection predicate based on a displacement from a finite element field.
   *
   * This class is intended for usage with parallel::shared::Triangulation. In
   * particular, the bounding boxes associated with all active cells will be
   * present on all processors. This is useful for creating an
   * OverlapTriangulation on each processor with bounding boxes intersecting an
   * arbitrary part of the Triangulation.
   */
  template <int dim, int spacedim = dim>
  class FEIntersectionPredicate : public IntersectionPredicate<dim, spacedim>
  {
  public:
    FEIntersectionPredicate(const std::vector<BoundingBox<spacedim>> &bboxes,
                            const MPI_Comm &                 communicator,
                            const DoFHandler<dim, spacedim> &dof_handler,
                            const Mapping<dim, spacedim> &   mapping)
      : tria(&dof_handler.get_triangulation())
      , patch_bboxes(bboxes)
    {
      // TODO: support multiple FEs
      const FiniteElement<dim> &fe = dof_handler.get_fe();
      // TODO: also check bboxes by position of quadrature points instead of
      // just nodes. Use QProjector to place points solely on cell boundaries.
      const Quadrature<dim> nodal_quad(fe.get_unit_support_points());

      FEValues<dim, spacedim> fe_values(mapping,
                                        fe,
                                        nodal_quad,
                                        update_quadrature_points);

      active_cell_bboxes.resize(
        dof_handler.get_triangulation().n_active_cells());
      for (const auto cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const BoundingBox<spacedim> dbox(fe_values.get_quadrature_points());
            BoundingBox<spacedim, float> fbox;
            fbox.get_boundary_points() = dbox.get_boundary_points();
            active_cell_bboxes[cell->active_cell_index()] = fbox;
          }

      // TODO: use rtrees in parallel so that we don't need every bbox on every
      // processor in this intermediate step
      constexpr auto n_floats_per_bbox = spacedim * 2;
      static_assert(sizeof(active_cell_bboxes[0]) ==
                      sizeof(float) * n_floats_per_bbox,
                    "packing failed");
      const auto size = n_floats_per_bbox * active_cell_bboxes.size();
      // TODO assert sizes are all equal and nonzero
      const int ierr =
        MPI_Allreduce(MPI_IN_PLACE,
                      reinterpret_cast<float *>(&active_cell_bboxes[0]),
                      size,
                      MPI_FLOAT,
                      MPI_SUM,
                      communicator);
      AssertThrowMPI(ierr);

      for (const auto &bbox : active_cell_bboxes)
        {
          Assert(bbox.volume() > 0, ExcMessage("bboxes should not be empty"));
        }
    }

    virtual bool
    operator()(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      const override
    {
      Assert(&cell->get_triangulation() == tria,
             ExcMessage("only valid for inputs constructed from the originally "
                        "provided Triangulation"));
      // If the cell is active check its bbox:
      if (cell->is_active())
        {
          const auto &cell_bbox = active_cell_bboxes[cell->active_cell_index()];
          for (const auto &bbox : patch_bboxes)
            if (intersects(cell_bbox, bbox))
              return true;
          return false;
        }
      // Otherwise see if it has a descendant that intersects:
      else if (cell->has_children())
        {
          const auto n_children             = cell->n_children();
          bool       has_intersecting_child = false;
          for (unsigned int child_n = 0; child_n < n_children; ++child_n)
            {
              const bool child_intersects = (*this)(cell->child(child_n));
              if (child_intersects)
                {
                  has_intersecting_child = true;
                  break;
                }
            }
          return has_intersecting_child;
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }

      Assert(false, ExcFDLInternalError());
      return false;
    }

    const SmartPointer<const Triangulation<dim, spacedim>> tria;
    const std::vector<BoundingBox<spacedim>>               patch_bboxes;
    std::vector<BoundingBox<spacedim, float>>              active_cell_bboxes;
  };
} // namespace fdl

#endif
