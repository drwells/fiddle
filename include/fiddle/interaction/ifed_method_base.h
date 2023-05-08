#ifndef included_fiddle_interaction_ifed_method_base_h
#define included_fiddle_interaction_ifed_method_base_h

#include <fiddle/base/config.h>

#include <ibamr/IBStrategy.h>

#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   */
  template <int dim, int spacedim = dim>
  class IFEDMethodBase : public IBAMR::IBStrategy
  {
    static_assert(spacedim == NDIM, "Only available for spacedim == NDIM");
  };
} // namespace fdl

#endif
