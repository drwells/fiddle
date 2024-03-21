#include <fiddle/base/samrai_utilities.h>

#include <fiddle/postprocess/volume_meter.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <PatchHierarchy.h>

#include <cmath>

namespace fdl
{
  template <int spacedim>
  VolumeMeter<spacedim>::VolumeMeter(
    const Point<spacedim>                        &center,
    const double                                 &radius,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : Meter<spacedim, spacedim>(patch_hierarchy)
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_volume_meter_ctor,
                              "fdl::VolumeMeter::VolumeMeter()");
    GridGenerator::hyper_ball_balanced(this->m_meter_tria, center, radius);
    const double dx = compute_min_cell_width(patch_hierarchy);

    // On average, make MFAC 1 by solving (r0 / 2) / 2^n = delta x for n. With
    // hyper_ball_balanced() there are about two points in the radial direction
    // when the mesh is initially created. Hence we want to refine n times so
    // that (r0 / 2) is ultimately equal to delta x.
    const auto n_refinements =
      static_cast<int>(std::ceil(std::log2(radius / dx) - 1.0));
    this->m_meter_tria.refine_global(std::max(0, n_refinements));
    Meter<spacedim, spacedim>::internal_reinit();
  }

  template <int spacedim>
  void
  VolumeMeter<spacedim>::reinit(const Point<spacedim> &new_center)
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_volume_meter_reinit,
                              "fdl::VolumeMeter::reinit()");
    GridTools::shift(new_center - this->get_centroid(), this->m_meter_tria);
    Meter<spacedim, spacedim>::internal_reinit();
  }

  template <int spacedim>
  void
  VolumeMeter<spacedim>::reinit()
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_volume_meter_reinit,
                              "fdl::VolumeMeter::reinit()");
    // special case: nothing can move so skip all but one reinit function
    this->reinit_interaction();
  }

  template class VolumeMeter<NDIM>;
} // namespace fdl
