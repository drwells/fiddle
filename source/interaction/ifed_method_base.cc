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
