#include <fiddle/interaction/ifed_method_base.h>

#include <limits>

namespace fdl
{
  template <int dim, int spacedim>
  IFEDMethodBase<dim, spacedim>::IFEDMethodBase(
    std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
    std::vector<Part<dim, spacedim>>     &&input_parts)
    : current_time(std::numeric_limits<double>::signaling_NaN())
    , half_time(std::numeric_limits<double>::signaling_NaN())
    , new_time(std::numeric_limits<double>::signaling_NaN())
    , parts(std::move(input_parts))
    , surface_parts(std::move(input_surface_parts))
    , part_vectors(this->parts)
    , surface_part_vectors(this->surface_parts)
  {}

  template class IFEDMethodBase<NDIM, NDIM>;
} // namespace fdl
