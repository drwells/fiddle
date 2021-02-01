#ifndef included_fiddle_interaction_interaction_h
#define included_fiddle_interaction_interaction_h

#include <fiddle/grid/patch_map.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/vector.h>

#include <PatchLevel.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(
    const std::vector<BoundingBox<spacedim, Number>> &        bboxes,
    const int                                                 tag_index,
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level);

  /**
   * Compute the right-hand side used to project the velocity from Eulerian to
   * Lagrangian representation.
   *
   * @param[in] data_idx the SAMRAI patch data index we are interpolating. The
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
   * @param[in] f_dof_handler DoFHandler for the finite element we are
   * interpolating onto.
   *
   * @param[in] f_mapping Mapping for computing values of the finite element
   * field on the reference configuration.
   *
   * @param[out] f_rhs The load vector populated by this operation.
   *
   * @note In general, an OverlappingTriangulation has no knowledge of whether
   * or not DoFs on its boundaries should be constrained. Hence information must
   * first be communicated between processes and then constraints should be
   * applied.
   */
  template <int dim, int spacedim = dim>
  void
  compute_projection_rhs(const int                        f_data_idx,
                         const PatchMap<dim, spacedim> &  patch_map,
                         const Mapping<dim, spacedim> &   position_mapping,
                         const std::vector<unsigned char> &  quadrature_indices,
                         const std::vector<Quadrature<dim>> &quadratures,
                         const DoFHandler<dim, spacedim> &f_dof_handler,
                         const Mapping<dim, spacedim> &   f_mapping,
                         Vector<double> &                 f_rhs);
} // namespace fdl
#endif
