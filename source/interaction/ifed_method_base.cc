#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/ifed_method_base.h>
#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <ibamr/ibamr_utilities.h>

#include <limits>

namespace
{
  using namespace SAMRAI;
  static tbox::Timer *t_apply_gradient_detector;
} // namespace

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
  {
    auto set_timer = [&](const char *name)
    { return tbox::TimerManager::getManager()->getTimer(name); };
    t_apply_gradient_detector =
      set_timer("fdl::IFEDMethodBase::applyGradientDetector()");
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::applyGradientDetector(
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
    int                                               level_number,
    double /*error_data_time*/,
    int tag_index,
    bool /*initial_time*/,
    bool /*uses_richardson_extrapolation_too*/)
  {
    IBAMR_TIMER_START(t_apply_gradient_detector);
    // TODO: we should find a way to save the bboxes so they do not need to be
    // computed for each level that needs tagging - conceivably this could
    // happen in beginDataRedistribution() and the array can be cleared in
    // endDataRedistribution()
    auto do_tag = [&](const auto &collection)
    {
      for (const auto &part : collection)
        {
          constexpr int structdim =
            std::remove_reference_t<decltype(collection[0])>::dimension;
          MappingFEField<structdim,
                         spacedim,
                         LinearAlgebra::distributed::Vector<double>>
                     mapping(part.get_dof_handler(), part.get_position());
          const auto local_bboxes =
            compute_cell_bboxes<structdim, spacedim, float>(
              part.get_dof_handler(), mapping);
          // Like most other things this only works with p::s::T now
          const auto &tria = dynamic_cast<
            const parallel::shared::Triangulation<structdim, spacedim> &>(
            part.get_triangulation());
          const auto global_bboxes =
            collect_all_active_cell_bboxes(tria, local_bboxes);
          tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
            hierarchy->getPatchLevel(level_number);
          Assert(patch_level, ExcNotImplemented());
          tag_cells(global_bboxes, tag_index, patch_level);
        }
    };

    do_tag(parts);
    do_tag(surface_parts);
    IBAMR_TIMER_STOP(t_apply_gradient_detector);
  }

  //
  // Time stepping
  //

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::forwardEulerStep(double current_time,
                                                  double new_time)
  {
    const double dt = new_time - current_time;
    Assert(this->current_time == current_time, ExcFDLNotImplemented());
    auto do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_position());
          new_position.set_ghost_state(false);
          new_position.add(dt, part.get_velocity());
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.set_ghost_state(false);
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(surface_parts, surface_part_vectors);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::backwardEulerStep(double current_time,
                                                   double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::midpointStep(double current_time,
                                              double new_time)
  {
    const double dt = new_time - current_time;
    Assert(this->current_time == current_time, ExcFDLNotImplemented());
    auto do_step = [&](auto &collection, auto &vectors)
    {
      for (unsigned int i = 0; i < collection.size(); ++i)
        {
          auto &part = collection[i];
          AssertThrow(vectors.dimension == part.dimension,
                      ExcFDLInternalError());
          // Set the position at the end time:
          LinearAlgebra::distributed::Vector<double> new_position(
            part.get_position());
          new_position.set_ghost_state(false);
          new_position.add(dt, vectors.get_velocity(i, half_time));
          vectors.set_position(i, new_time, std::move(new_position));

          // Set the position at the half time:
          LinearAlgebra::distributed::Vector<double> half_position(
            part.get_partitioner());
          half_position.set_ghost_state(false);
          half_position.add(0.5,
                            vectors.get_position(i, current_time),
                            0.5,
                            vectors.get_position(i, new_time));
          vectors.set_position(i, half_time, std::move(half_position));
        }
    };
    do_step(parts, part_vectors);
    do_step(surface_parts, surface_part_vectors);
  }

  template <int dim, int spacedim>
  void
  IFEDMethodBase<dim, spacedim>::trapezoidalStep(double current_time,
                                                 double new_time)
  {
    (void)current_time;
    (void)new_time;
    Assert(false, ExcFDLNotImplemented());
  }


  template class IFEDMethodBase<NDIM, NDIM>;
} // namespace fdl
