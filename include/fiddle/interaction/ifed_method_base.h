#ifndef included_fiddle_interaction_ifed_method_base_h
#define included_fiddle_interaction_ifed_method_base_h

#include <fiddle/base/config.h>

#include <ibamr/IBStrategy.h>

#include <vector>

namespace fdl
{
  /**
   */
  template <int dim, int spacedim = dim>
  class IFEDMethodBase : public IBAMR::IBStrategy
  {
  public:
    static_assert(spacedim == NDIM, "Only available for spacedim == NDIM");

    IFEDMethodBase();

  protected:
    double current_time;
    double half_time;
    double new_time;
  };
} // namespace fdl

#endif
