#include <fiddle/interaction/ifed_method_base.h>

#include <limits>

namespace fdl
{
  template <int dim, int spacedim>
  IFEDMethodBase<dim, spacedim>::IFEDMethodBase()
    : current_time(std::numeric_limits<double>::signaling_NaN())
    , half_time(std::numeric_limits<double>::signaling_NaN())
    , new_time(std::numeric_limits<double>::signaling_NaN())
  {}

  template class IFEDMethodBase<NDIM, NDIM>;
} // namespace fdl
