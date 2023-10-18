#ifndef included_fiddle_interaction_interaction_utilities_h
#define included_fiddle_interaction_interaction_utilities_h

#include <fiddle/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/std_cxx17/optional.h>

#include <memory>
#include <vector>

// forward declarations
namespace fdl
{
  template <int, int>
  class NodalPatchMap;
  template <int, int>
  class PatchMap;
} // namespace fdl

namespace dealii
{
  template <int, int>
  class DoFHandler;
  template <int, int>
  class Mapping;
  template <typename>
  class Vector;
} // namespace dealii

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class PatchLevel;
  }

  namespace tbox
  {
    template <typename>
    class Pointer;
  }
} // namespace SAMRAI

// This file contains the functions that do all the actual interaction work -
// these are typically called by InteractionBase and its descendants and not
// directly by user code.

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(const std::vector<BoundingBox<spacedim, Number>> &bboxes,
            const int                                         tag_index,
            tbox::Pointer<hier::PatchLevel<spacedim>>        &patch_level);

  /**
   * Add the number of quadrature points.
   *
   * @param[in] qp_data_index the SAMRAI patch data index - the values in the
   * cells will be set to the number of quadrature points intersecting that
   * cell. The corresponding variable should be cell-centered, have a depth of
   * 1, and have either int, float, or double type.
   *
   * @param[in] patch_map The mapping between SAMRAI patches and deal.II cells
   * which we will use for counting quadrature points. This is logically not
   * const because we need to modify the SAMRAI data accessed through a pointer
   * owned by this class.
   *
   * @param[in] position_mapping Mapping from the reference configuration to the
   * current configuration of the mesh.
   *
   * @param[in] quadrature_indices This vector is indexed by the active cell
   * index - the value is the index into @p quadratures corresponding to the
   * correct quadrature rule on that cell.
   *
   * @param[in] quadratures The vector of quadratures we use for interaction.
   *
   * @note This is a purely local operation since we always assume a PatchMap
   * stores every element that intersects with the interior of a patch.
   */
  template <int dim, int spacedim = dim>
  void
  count_quadrature_points(const int                         qp_data_index,
                          PatchMap<dim, spacedim>          &patch_map,
                          const Mapping<dim, spacedim>     &position_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<dim>> &quadratures);

  /**
   * Count the number of nodes in each patch.
   *
   * @param[in] node_count_data_index Data index into which we will add the
   * number of nodes in each cell.
   *
   * @param[in] nodal_patch_map Mapping between patches and DoFs.
   *
   * @param[in] position Nodal coordinates in node-first ordering.
   */
  template <int dim, int spacedim>
  void
  count_nodes(const int                     node_count_data_index,
              NodalPatchMap<dim, spacedim> &nodal_patch_map,
              const Vector<double>         &position);

  /**
   * Compute the right-hand side used to project the velocity from Eulerian to
   * Lagrangian representation.
   *
   * @param[in] data_index the SAMRAI patch data index we are interpolating. The
   * depth of the variable must match the number of components of the finite
   * element.
   *
   * @param[in] patch_map The mapping between SAMRAI patches and deal.II cells
   * which we will use for interpolation.
   *
   * @param[in] position_mapping Mapping from the reference configuration to the
   * current configuration of the mesh.
   *
   * @param[in] quadrature_indices This vector is indexed by the active cell
   * index - the value is the index into @p quadratures corresponding to the
   * correct quadrature rule on that cell.
   *
   * @param[in] quadratures The vector of quadratures we use to interpolate.
   *
   * @param[in] dof_handler DoFHandler for the finite element we are
   * interpolating onto.
   *
   * @param[in] mapping Mapping for computing values of the finite element
   * field on the reference configuration.
   *
   * @param[out] rhs The load vector populated by this operation.
   *
   * @note In general, an OverlappingTriangulation has no knowledge of whether
   * or not DoFs on its boundaries should be constrained. Hence information must
   * first be communicated between processes and then constraints should be
   * applied.
   */
  template <int dim, int spacedim = dim>
  void
  compute_projection_rhs(const std::string                  &kernel_name,
                         const int                           data_index,
                         const PatchMap<dim, spacedim>      &patch_map,
                         const Mapping<dim, spacedim>       &position_mapping,
                         const std::vector<unsigned char>   &quadrature_indices,
                         const std::vector<Quadrature<dim>> &quadratures,
                         const DoFHandler<dim, spacedim>    &dof_handler,
                         const Mapping<dim, spacedim>       &mapping,
                         Vector<double>                     &rhs);

  /**
   * Interpolate Eulerian data at specified Lagrangian points.
   *
   * @param[in] data_index the SAMRAI patch data index we are interpolating. The
   * depth of the variable must match the number of components of the Lagrangian
   * data (which is implicitly specified by the length of @p interpolated_values).
   *
   * @param[in] patch_map The mapping between SAMRAI patches and points which we
   * will use for interpolation.
   *
   * @param[in] position The vector containing the positions of each Lagrangian
   * point in node-first ordering (i.e., the vector contains {x0, y0, x1, y1,
   * ...}). The number of points is implicitly encoded by the length of the
   * vector and `spacedim`.
   *
   * @param[out] interpolated_values Vector of values interpolated at each node.
   *
   * @note While this function does not directly use any finite element data
   * structures (such as a DoFHandler or FiniteElement), it does assume that we
   * use a FE-like numbering of the DoFs: i.e., each component of the position
   * at each node is assigned a unique DoF index in the typical way. This
   * information is provided in @p patch_map.
   */
  template <int dim, int spacedim>
  void
  compute_nodal_interpolation(const std::string                  &kernel_name,
                              const int                           data_index,
                              const NodalPatchMap<dim, spacedim> &patch_map,
                              const Vector<double>               &position,
                              Vector<double> &interpolated_values);

  /**
   * Compute (by adding into the patch index @p data_index) the forces on the
   * Eulerian grid corresponding to the Lagrangian field F.
   *
   * @param[in] data_index the SAMRAI patch data index into which we are
   * spreading. The depth of the variable must match the number of components of
   * the finite element.
   *
   * @param[inout] patch_map The mapping between SAMRAI patches and deal.II
   * cells. Though we do not modify this object directly, it is logically
   * non-const because we will modify the patches owned by the patch hierarchy
   * to which this object stores pointers.
   *
   * @param[in] position_mapping Mapping from the reference configuration to the
   * current configuration of the mesh.
   *
   * @param[in] quadrature_indices This vector is indexed by the active cell
   * index - the value is the index into @p quadratures corresponding to the
   * correct quadrature rule on that cell.
   *
   * @param[in] quadratures The vector of quadratures we use to interpolate.
   *
   * @param[in] dof_handler DoFHandler for the finite element we are
   * spreading from.
   *
   * @param[in] mapping Mapping for computing values of the finite element
   * field on the reference configuration.
   *
   * @param[in] solution The finite element field we are spreading from.
   */
  template <int dim, int spacedim>
  void
  compute_spread(const std::string                  &kernel_name,
                 const int                           data_index,
                 PatchMap<dim, spacedim>            &patch_map,
                 const Mapping<dim, spacedim>       &position_mapping,
                 const std::vector<unsigned char>   &quadrature_indices,
                 const std::vector<Quadrature<dim>> &quadratures,
                 const DoFHandler<dim, spacedim>    &dof_handler,
                 const Mapping<dim, spacedim>       &mapping,
                 const Vector<double>               &solution);

  /**
   * Spread Lagrangian data at specified Lagrangian points.
   *
   * @param[in] data_index the SAMRAI patch data index into which we spread. The
   * depth of the variable must match the number of components of the Lagrangian
   * data (which is implicitly specified by the length of @p spread_values).
   *
   * @param[inout] patch_map The mapping between SAMRAI patches and points at
   * which we spread values.
   *
   * @param[in] position The vector containing the positions of each Lagrangian
   * point in node-first ordering (i.e., the vector contains {x0, y0, x1, y1,
   * ...}). The number of points is implicitly encoded by the length of the
   * vector and `spacedim`.
   *
   * @param[in] spread_values Vector of values we spread.
   *
   * @note While this function does not directly use any finite element data
   * structures (such as a DoFHandler or FiniteElement), it does assume that we
   * use a FE-like numbering of the DoFs: i.e., each component of the position
   * at each node is assigned a unique DoF index in the typical way. This
   * information is provided in @p patch_map.
   */
  template <int dim, int spacedim>
  void
  compute_nodal_spread(const std::string            &kernel_name,
                       const int                     data_index,
                       NodalPatchMap<dim, spacedim> &patch_map,
                       const Vector<double>         &position,
                       const Vector<double>         &spread_values);

  /**
   * Compute intersection the point of a line with an edge.
   *
   * This function returns the convex combination coefficient of the
   * intersection between the start of the FD stencil and its end. Null pointer
   * would be returned if there isn't an intersection.
   *
   * Linear interpolation:
   *
   * 0.5*(1-u)*p0 + 0.5*(1+u)*p1 = r + t*q
   *
   * Factor the interpolation formula:
   *
   * 0.5*(p1-p0)*u+0.5*(p1+p0) = r + t*q
   *
   * Solve for u:
   *
   *    a*u + b = 0
   *
   * with:
   *
   *   a = 0.5*(-p0+p1)
   *   b = 0.5*(p0+p1) - r
   *
   * @param[in] simplex an array containing Eulerian location of
   * vertices of the given element.
   *
   * @param[in] stencil_start the center location of the FD stencil.
   *
   * @param[in] stencil_width the width of the given FD stencil.
   *
   * @param[in] stencil_axis a given index indicate the stencil axis which
   * suppose to be aligned with the coordinate axes.
   *
   * @note This function is only work for P1 element.
   */
  std_cxx17::optional<double>
  intersect_line_with_edge(const std::array<Point<2>, 2> &simplex,
                           const Point<2>                &stencil_start,
                           const double                  &stencil_width,
                           const unsigned int             stencil_axis);


  /**
   * Compute intersection the point of a line with a flat triangle
   *
   * This function returns the convex combination coefficient of the
   * intersection between the start of the FD stencil and its end. Null pointer
   * would be returned if there isn't an intersection.
   *
   * Due to large range of potential values of MFAC (interface(or
   * structure)_mesh_size/fluid_mesh_size) We first run a simple check if the
   * given line would intersect with a "minimum" box contains the triangle. Then
   * we solve for intersections with
   *
   * Linear interpolation:
   *
   * (1-u-v)*p0 + u*p1 + v*p2 = p + t*d
   *
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   * http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
   * At the end we check if the intersection is within the projected triangle.
   *
   * @param[in] simplex an array containing Eulerian location of
   * vertices of the given element.
   *
   * @param[in] stencil_start the center location of the FD stencil.
   *
   * @param[in] stencil_width the width of the given FD stencil.
   *
   * @param[in] stencil_axis a given index indicate the stencil axis which
   * suppose to be aligned with the coordinate axes.
   *
   * @note This function is only work for P1 element.
   */
  std_cxx17::optional<double>
  intersect_line_with_flat_triangle(const std::array<Point<3>, 3> &simplex,
                                    const Point<3>    &stencil_start,
                                    const double      &stencil_width,
                                    const unsigned int stencil_axis);

  /**
   * Compute intersection the point of a line with a simplex.
   *
   * @param[in] simplex an array containing Eulerian location of
   * vertices of the given element.
   *
   * @param[in] stencil_start the center location of the FD stencil.
   *
   * @param[in] stencil_width the width of the given FD stencil.
   *
   * @param[in] stencil_axis a given index indicate the stencil axis which
   * suppose to be aligned with the coordinate axes.
   *
   */
  template <int spacedim>
  std_cxx17::optional<double>
  intersect_stencil_with_simplex(
    const std::array<Point<spacedim + 1>, spacedim + 1> &simplex,
    const Point<spacedim + 1>                           &stencil_start,
    const double                                        &stencil_width,
    const unsigned int                                   stencil_axis);

} // namespace fdl
#endif
