#include <fiddle/base/samrai_utilities.h>

#include <fiddle/postprocess/volume_meter.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <PatchHierarchy.h>

namespace fdl
{
  template <int spacedim>
  VolumeMeter<spacedim>::VolumeMeter(
    const Point<spacedim>                        &center,
    const double                                 &radius,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : MeterBase<spacedim, spacedim>(patch_hierarchy)
    , center(center)
  {
    GridGenerator::hyper_ball_balanced(this->meter_tria, center, radius);
    const double dx = compute_min_cell_width(patch_hierarchy);

    while (GridTools::maximal_cell_diameter(this->meter_tria) > dx)
      this->meter_tria.refine_global(1);
    MeterBase<spacedim, spacedim>::internal_reinit();
  }

  template <int spacedim>
  void
  VolumeMeter<spacedim>::reinit(const Point<spacedim> &new_center)
  {
    const Tensor<1, spacedim> displacement = new_center - center;
    GridTools::shift(displacement, this->meter_tria);
    center = new_center;

    MeterBase<spacedim, spacedim>::internal_reinit();
  }

  template <int spacedim>
  void
  VolumeMeter<spacedim>::reinit()
  {
    // special case: nothing can move so skip all but one reinit function
    this->reinit_interaction();
  }

  template class VolumeMeter<NDIM>;
} // namespace fdl