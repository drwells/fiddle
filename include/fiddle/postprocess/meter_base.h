#ifndef included_fiddle_postprocess_meter_base_h
#define included_fiddle_postprocess_meter_base_h

#include <fiddle/base/config.h>

#include <deal.II/distributed/shared_tria.h>

#include <tbox/Pointer.h>

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class PatchHierarchy;
  }
} // namespace SAMRAI


namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Base class for the meter classes (SurfaceMeter and VolumeMeter).
   */
  template <int dim, int spacedim = dim>
  class MeterBase
  {
  public:
    /**
     * Constructor.
     */
    MeterBase(tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy);

    /* @name object access
     * @{
     */

    /**
     * Return a reference to the meter Triangulation. This triangulation is
     * not in reference coordinates: instead its absolute position is
     * determined by the position vector specified to the constructor or
     * reinit().
     */
    const Triangulation<dim, spacedim> &
    get_triangulation() const;

    /** @} */

    /* @name FSI
     * @{
     */

    /**
     * Return whether or not all vertices of the Triangulation are actually
     * inside the domain defined by the PatchHierarchy.
     */
    bool
    compute_vertices_inside_domain() const;

    /** @} */

  protected:
    /**
     * Cartesian-grid data.
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy;

    /**
     * Meter Triangulation.
     */
    parallel::shared::Triangulation<dim, spacedim> meter_tria;
  };


  // --------------------------- inline functions --------------------------- //


  template <int dim, int spacedim>
  inline const Triangulation<dim, spacedim> &
  MeterBase<dim, spacedim>::get_triangulation() const
  {
    return meter_tria;
  }
} // namespace fdl

#endif
