#ifndef included_fiddle_interaction_interaction_utilities_h
#define included_fiddle_interaction_interaction_utilities_h

#include <fiddle/base/config.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>

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
   * Compute intersection the point of a line with an edge
   *
   * WARNING: This code is specialized to the case in which q is a unit vector
   * aligned with the coordinate axes.
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
   * @param[out] t_vals The vector containing the intersection point in Eulerian coordinate
   * and element reference coordinate.
   *
   * @param[in] element_vertices The array containing Eulerian location of vertices of the given element.
   *
   * @param[in] r The center location of the FD stencil`.
   *
   * @param[in] q a unit vector aligned with the coordinate axes.
   *
   * @param[in] tol a given tolerance for deciding if the intersection point is within the edge.
   *
   * @note This function is only work for P1 element.
   */
  void
  intersect_line_with_edge(std::vector<std::pair<double, Point<1>> >& t_vals,
                           std::array<Point<2>, 2> element_vertices,
                           dealii::Point<2> r,
                           dealii::Tensor<1,2> q,
                           const double tol);

    /**
   * This code is used to determine if the given line would intersect with the triangle
   * Also perturb r if the FD stencil cutting boundary of the triangle.
   *
   * https://mathworld.wolfram.com/TriangleInterior.html
   *
   * @param[in] elements_vertices The array containing Eulerian location of vertices of the given element.
   *
   * @param[in] r The center location of the FD stencil`.
   *
   * @param[in] q a unit vector aligned with the coordinate axes.
   *
   * @param[in] tol a given tolerance for deciding if the intersection point is within the edge.
   */
  template <int dim, int spacedim>
  bool
  if_line_intersect_with_triangle(const std::array<Point<spacedim>, spacedim>& element_vetices, dealii::Point<spacedim>& r,
                                  const dealii::Tensor<1,spacedim>& q);

  /**
 * This code is used to determine if the given line would intersect with a rectangle containing the triangle
 *
 * @param[in] elements_vertices The array containing Eulerian location of vertices of the given element.
 *
 * @param[in] r The center location of the FD stencil`.
 *
 * @param[in] q a unit vector aligned with the coordinate axes.
 *
 * @param[in] tol a given tolerance for deciding if the intersection point is within the edge.
 */
  template <int dim, int spacedim>
  bool
  if_line_intersect_with_box(std::array<Point<spacedim>, spacedim>& element_vetices, dealii::Point<spacedim>& r,
                             dealii::Tensor<1,spacedim> q);
/**
  * Compute intersection the point of a line with a flat triangle
  *
  * WARNING: This code is specialized to the case in which q is a unit vector
  * aligned with the coordinate axes.
  *
  * Linear interpolation:
  *
  * (1-u-v)*p0 + u*p1 + v*p2 = p + t*d
  *
  * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
  *
  * @param[out] t_vals The vector containing the intersection point in Eulerian coordinate
  * and element reference coordinate.
  *
  * @param[in] element_vertices The array containing Eulerian location of vertices of the given element.
  *
  * @param[in] r The center location of the FD stencil`.
  *
  * @param[in] q a unit vector aligned with the coordinate axes.
  *
  * @param[in] tol a given tolerance for deciding if the intersection point is within the edge.
  *
  * @note This function is only work for P1 element.
  */
  void
  intersect_line_with_flat_triangle(std::vector<std::pair<double, Point<2>> >& t_vals,
                                           std::array<Point<3>, 3> element_vetices,
                                           dealii::Point<3> r,
                                           const dealii::Tensor<1,3> q,
                                           const double tol = 0);
/**
* Compute intersection the point of a line with an element.
*
* WARNING: This code is specialized to the case in which q is a unit vector
* aligned with the coordinate axes.
*
* @param[out] t_vals The vector containing the intersection point in Eulerian coordinate
* and element reference coordinate.
*
* @param[in] element_vertices The array containing Eulerian location of vertices of the given element.
*
* @param[in] r The center location of the FD stencil`.
*
* @param[in] q a unit vector aligned with the coordinate axes.
*
* @param[in] tol a given tolerance for deciding if the intersection point is within the edge.
*
* @note This function is only work for P1 element.
*/
  template <int dim, int spacedim>
  void
  intersect_line_with_element(std::vector<std::pair<double, Point<dim>> >& t_vals,
                                std::array<Point<spacedim>, spacedim> element_vetices,
                                dealii::Point<spacedim> r,
                                dealii::Tensor<1,spacedim> q,
                                const double tol);

} // namespace fdl
#endif
