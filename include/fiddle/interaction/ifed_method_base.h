#ifndef included_fiddle_interaction_ifed_method_base_h
#define included_fiddle_interaction_ifed_method_base_h

#include <fiddle/base/config.h>

#include <fiddle/mechanics/part.h>
#include <fiddle/mechanics/part_vectors.h>

#include <ibamr/IBStrategy.h>

#include <BasePatchHierarchy.h>
#include <tbox/Pointer.h>

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
  public:
    static_assert(spacedim == NDIM, "Only available for spacedim == NDIM");

    IFEDMethodBase(std::vector<Part<dim - 1, spacedim>> &&input_surface_parts,
                   std::vector<Part<dim, spacedim>>     &&input_parts);
    /**
     * @name fluid-structure interaction.
     * @{
     */

    /**
     * Tag cells in @p hierarchy that intersect with the structure.
     */
    virtual void
    applyGradientDetector(
      tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
      int                                               level_number,
      double                                            error_data_time,
      int                                               tag_index,
      bool                                              initial_time,
      bool uses_richardson_extrapolation_too) override;
    /**
     * @}
     */

    /**
     * @name timestepping.
     * @{
     */
    virtual void
    forwardEulerStep(double current_time, double new_time) override;

    virtual void
    backwardEulerStep(double current_time, double new_time) override;

    virtual void
    midpointStep(double current_time, double new_time) override;

    virtual void
    trapezoidalStep(double current_time, double new_time) override;
    /**
     * @}
     */

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
