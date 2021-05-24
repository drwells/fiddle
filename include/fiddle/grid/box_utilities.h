#ifndef included_fiddle_grid_box_utilities_h
#define included_fiddle_grid_box_utilities_h

#include <fiddle/base/exceptions.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <BasePatchLevel.h>
#include <Patch.h>
#include <PatchLevel.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Fast intersection check between two bounding boxes.
   */
  template <int spacedim, typename Number1, typename Number2>
  bool
  intersects(const BoundingBox<spacedim, Number1> &a,
             const BoundingBox<spacedim, Number2> &b);

  /**
   * Compute the bounding boxes for a set of SAMRAI patches. In addition, if
   * necessary, expand each bounding box by @p extra_ghost_cell_fraction times
   * the length of a cell in each coordinate direction.
   */
  template <int spacedim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_patch_bboxes(
    const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
      &          patches,
    const double extra_ghost_cell_fraction = 0.0);

  /**
   * Compute the bounding boxes for all locally owned and active cells for a
   * finite element field.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  compute_cell_bboxes(const DoFHandler<dim, spacedim> &dof_handler,
                      const Mapping<dim, spacedim> &   mapping)
  {
    // TODO: support multiple FEs
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();
    // TODO: also check bboxes by position of quadrature points instead of
    // just nodes. Use QProjector to place points solely on cell boundaries.
    const Quadrature<dim> nodal_quad(fe.get_unit_support_points());

    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      nodal_quad,
                                      update_quadrature_points);

    std::vector<BoundingBox<spacedim, Number>> bboxes;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          const BoundingBox<spacedim> dbox(fe_values.get_quadrature_points());
          // we have to do a conversion if Number != double
          BoundingBox<spacedim, Number> fbox;
          fbox.get_boundary_points() = dbox.get_boundary_points();
          bboxes.push_back(fbox);
        }
    return bboxes;
  }

  /**
   * Collect all bounding boxes on all processors.
   */
  template <int dim, int spacedim = dim, typename Number = double>
  std::vector<BoundingBox<spacedim, Number>>
  collect_all_active_cell_bboxes(
    const parallel::shared::Triangulation<dim, spacedim> &tria,
    const std::vector<BoundingBox<spacedim, Number>> &local_active_cell_bboxes)
  {
    Assert(tria.n_active_cells() == local_active_cell_bboxes.size(),
           ExcMessage("There should be a local bbox for each local active cell"));
    std::vector<BoundingBox<spacedim, Number>> global_bboxes(
      tria.n_active_cells());
    unsigned int cell_n = 0;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          AssertIndexRange(cell->active_cell_index(), local_active_cell_bboxes.size());
          global_bboxes[cell->active_cell_index()] =
            local_active_cell_bboxes[cell_n];
          ++cell_n;
        }
    Assert(cell_n == local_active_cell_bboxes.size(),
           ExcMessage("There should be a bounding box for every locally owned cell"));

    MPI_Datatype   mpi_type  = {};
    constexpr bool is_float  = std::is_same<Number, float>::value;
    constexpr bool is_double = std::is_same<Number, double>::value;
    static_assert(is_float || is_double, "Must be float or double");
    if (is_float)
      {
        mpi_type = MPI_FLOAT;
      }
    else if (is_double)
      {
        mpi_type = MPI_DOUBLE;
      }
    else
      {
        AssertThrow(false, ExcFDLNotImplemented());
      }

    constexpr auto n_nums_per_bbox = spacedim * 2;
    static_assert(sizeof(BoundingBox<spacedim, Number>) ==
                    sizeof(Number) * n_nums_per_bbox,
                  "packing failed");
    const auto size = n_nums_per_bbox * global_bboxes.size();
    // TODO MPI_Allgather would be MUCH better here
    const int ierr =
      MPI_Allreduce(MPI_IN_PLACE,
                    reinterpret_cast<Number *>(&global_bboxes[0]),
                    size,
                    mpi_type,
                    MPI_SUM,
                    tria.get_communicator());
    AssertThrowMPI(ierr);

    for (const auto &bbox : global_bboxes)
      {
        Assert(bbox.volume() > 0, ExcMessage("bboxes should not be empty"));
      }
    return global_bboxes;
  }

  /**
   * Helper function for extracting locally owned patches from a base patch
   * level.
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<spacedim>>
                    base_patch_level);

  /**
   * Helper function for extracting locally owned patches from a patch level.
   */
  template <int spacedim>
  std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
  extract_patches(
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level);


  // --------------------------- inline functions --------------------------- //


  template <int spacedim, typename Number1, typename Number2>
  inline bool
  intersects(const BoundingBox<spacedim, Number1> &a,
             const BoundingBox<spacedim, Number2> &b)
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
        if (!((a.lower_bound(d) <= b.lower_bound(d) &&
               b.lower_bound(d) <= a.upper_bound(d)) ||
              (a.lower_bound(d) <= b.upper_bound(d) &&
               b.upper_bound(d) <= a.upper_bound(d))) &&
            !((b.lower_bound(d) <= a.lower_bound(d) &&
               a.lower_bound(d) <= b.upper_bound(d))))
          {
            return false;
          }
      }
    return true;
  }
} // namespace fdl

#endif
