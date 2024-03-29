#ifndef included_fiddle_grid_nodal_patch_map_h
#define included_fiddle_grid_nodal_patch_map_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>

FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <Patch.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Data structure storing a compressed mapping between patches and nodal
   * DoF indices.
   *
   * This data structure assumes that we are using the new compressed nodal
   * representation available in DoFRenumbering::compute_support_point_wise(),
   * i.e., position (or velocity) DoFs at a single support point have adjacent
   * DoF indices and are sorted by vector component.
   */
  template <int dim, int spacedim = dim>
  class NodalPatchMap
  {
  public:
    /**
     * Default constructor. Sets up an empty mapping.
     */
    NodalPatchMap() = default;

    /**
     * Constructor. Associates DoFs to patches from provided coordinates.
     *
     * Here @p nodal_coordinates contains the (x, y, z) positions of nodes in
     * the order described in the general class documentation. This should be an
     * FE position vector or, more generally, just the positions of support
     * points. In particular, if a finite element field has multiple components
     * with equal support points (e.g., FESystem<2>(FE_Q<2>(1), 2)) then those
     * support points should not be duplicated in @p nodal_coordinates.
     *
     * Put another way: this class assumes that @p nodal_coordinates is the
     * coefficient vector of an FE field representing the position of a
     * structure. If this vector has N entries, then the IndexSet objects set up
     * by this class will enumerate dofs in the range [0, N - 1].
     *
     * The hier::Patch objects in @p patches may come from multiple patch
     * levels. If a node is present on multiple levels then it will be assigned
     * to the finest one.
     */
    NodalPatchMap(
      const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
      const std::vector<std::vector<BoundingBox<spacedim>>>   &patch_bboxes,
      const Vector<double> &nodal_coordinates);

    /**
     * Same as the constructor.
     */
    void
    reinit(const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
           const std::vector<std::vector<BoundingBox<spacedim>>> &patch_bboxes,
           const Vector<double> &nodal_coordinates);

    /**
     * Return the number of patches.
     */
    std::size_t
    size() const;

    /**
     * Access operator - returns an IndexSet containing the DoFs intersecting
     * either the patch or the patch's ghost area (as of the last regridding).
     */
    std::pair<const IndexSet &, tbox::Pointer<hier::Patch<spacedim>>>
    operator[](const std::size_t i);

    /**
     * Constant access operator.
     */
    std::pair<const IndexSet &, const tbox::Pointer<hier::Patch<spacedim>>>
    operator[](const std::size_t i) const;

  protected:
    // Patches.
    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;

    // For each patch, store the DoF indices which intersect (including the
    // extra ghost cell fraction) that patch.
    std::vector<IndexSet> patch_dof_indices;
  };


  // --------------------------- inline functions --------------------------- //


  template <int dim, int spacedim>
  std::size_t
  NodalPatchMap<dim, spacedim>::size() const
  {
    return patches.size();
  }

  template <int dim, int spacedim>
  std::pair<const IndexSet &, tbox::Pointer<hier::Patch<spacedim>>>
  NodalPatchMap<dim, spacedim>::operator[](const std::size_t i)
  {
    AssertIndexRange(i, size());
    return {patch_dof_indices[i], patches[i]};
  }

  template <int dim, int spacedim>
  std::pair<const IndexSet &, const tbox::Pointer<hier::Patch<spacedim>>>
  NodalPatchMap<dim, spacedim>::operator[](const std::size_t i) const
  {
    AssertIndexRange(i, size());
    return {patch_dof_indices[i], patches[i]};
  }
} // namespace fdl

#endif
