#ifndef included_fiddle_interaction_ifed_method_base_h
#define included_fiddle_interaction_ifed_method_base_h

#include <fiddle/base/config.h>

#include <fiddle/mechanics/part.h>
#include <fiddle/mechanics/part_vectors.h>

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

    IFEDMethodBase(std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
                   std::vector<Part<dim, spacedim>>     &&input_parts);

  protected:
    /**
     * Book-keeping
     * @{
     */
    double current_time;
    double half_time;
    double new_time;
    /**
     * @}
     */

    /**
     * Finite element data structures
     * @{
     */
    std::vector<Part<dim, spacedim>> parts;

    std::vector<Part<dim - 1, spacedim>> surface_parts;

    PartVectors<dim, spacedim> part_vectors;

    PartVectors<dim - 1, spacedim> surface_part_vectors;
    /**
     * @}
     */
  };
} // namespace fdl

#endif
