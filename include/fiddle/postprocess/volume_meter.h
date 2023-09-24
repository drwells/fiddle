#ifndef included_fiddle_postprocess_volume_meter_h
#define included_fiddle_postprocess_volume_meter_h

#include <fiddle/base/config.h>

#include <fiddle/postprocess/meter.h>

#include <deal.II/base/point.h>

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

  /**
   * Class for integrating Cartesian-grid values on a codimension zero
   * structure. Essentially the same as SurfaceMeter but without the flux parts.
   */
  template <int spacedim>
  class VolumeMeter : public Meter<spacedim, spacedim>
  {
  public:
    /**
     * Constructor. Sets up a sphere with approximately MFAC = 1 around @p
     * center with radius @p radius.
     */
    VolumeMeter(const Point<spacedim>                        &center,
                const double                                 &radius,
                tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy);

    /**
     * Reinitialize the volume meter to have a new center.
     */
    void
    reinit(const Point<spacedim> &new_center);

    /**
     * Alternative reinitialization function which only updates the internal
     * data structures to account for the PatchHierarchy being regridded.
     */
    void
    reinit();

  protected:
    Point<spacedim> center;
  };
} // namespace fdl

#endif
