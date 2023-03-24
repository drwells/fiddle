#include <fiddle/mechanics/part_vectors.h>

namespace fdl
{
  //
  // book keeping
  //
  template <int dim, int spacedim>
  PartVectors<dim, spacedim>::PartVectors(
    const std::vector<Part<dim, spacedim>> &parts)
  {
    for (const auto &part : parts)
      this->parts.push_back(&part);
  }



  template <int dim, int spacedim>
  void
  PartVectors<dim, spacedim>::begin_time_step(const double current_time,
                                              const double new_time)
  {
    this->current_time = current_time;
    this->new_time     = new_time;
    this->half_time    = current_time + 0.5 * (new_time - current_time);
  }



  template <int dim, int spacedim>
  void
  PartVectors<dim, spacedim>::end_time_step()
  {
    current_time = std::numeric_limits<double>::signaling_NaN();
    half_time    = std::numeric_limits<double>::signaling_NaN();
    new_time     = std::numeric_limits<double>::signaling_NaN();

    half_positions.clear();
    new_positions.clear();
    half_velocities.clear();
    new_velocities.clear();

    current_forces.clear();
    half_forces.clear();
    new_forces.clear();
  }



  template <int dim, int spacedim>
  std::vector<LinearAlgebra::distributed::Vector<double>>
  PartVectors<dim, spacedim>::get_all_new_positions()
  {
    Assert(parts.size() == new_positions.size(), ExcVectorNotAvailable());
    return std::move(new_positions);
  }



  template <int dim, int spacedim>
  std::vector<LinearAlgebra::distributed::Vector<double>>
  PartVectors<dim, spacedim>::get_all_new_velocities()
  {
    Assert(parts.size() == new_velocities.size(), ExcVectorNotAvailable());
    return std::move(new_velocities);
  }



  template <int dim, int spacedim>
  typename PartVectors<dim, spacedim>::TimeStep
  PartVectors<dim, spacedim>::get_time_step(const double time) const
  {
    if (std::abs(time - current_time) < 1e-12)
      return TimeStep::Current;
    if (std::abs(time - half_time) < 1e-12)
      return TimeStep::Half;
    if (std::abs(time - new_time) < 1e-12)
      return TimeStep::New;

    Assert(false, ExcFDLInternalError());
    return TimeStep::Current;
  }



  //
  // Vector access
  //
  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  PartVectors<dim, spacedim>::get_position(const unsigned int part_n,
                                           const double       time) const
  {
    AssertIndexRange(part_n, parts.size());
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          return parts[part_n]->get_position();
        case TimeStep::Half:
          Assert(part_n < half_positions.size(), ExcVectorNotAvailable());
          return half_positions[part_n];
        case TimeStep::New:
          Assert(part_n < new_positions.size(), ExcVectorNotAvailable());
          return new_positions[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n]->get_position();
  }



  template <int dim, int spacedim>
  void
  PartVectors<dim, spacedim>::set_position(
    const unsigned int                         part_n,
    const double                               time,
    LinearAlgebra::distributed::Vector<double> position)
  {
    AssertIndexRange(part_n, parts.size());
    position.set_ghost_state(false);
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          Assert(false, ExcMessage("cannot set position at current time"));
        case TimeStep::Half:
          half_positions.resize(
            std::max(std::size_t(part_n + 1), half_positions.size()));
          half_positions[part_n].swap(position);
          return;
        case TimeStep::New:
          new_positions.resize(
            std::max(std::size_t(part_n + 1), new_positions.size()));
          new_positions[part_n].swap(position);
          return;
      }

    Assert(false, ExcFDLInternalError());
  }



  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  PartVectors<dim, spacedim>::get_velocity(const unsigned int part_n,
                                           const double       time) const
  {
    AssertIndexRange(part_n, parts.size());
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          return parts[part_n]->get_velocity();
        case TimeStep::Half:
          Assert(part_n < half_velocities.size(), ExcVectorNotAvailable());
          return half_velocities[part_n];
        case TimeStep::New:
          Assert(part_n < new_velocities.size(), ExcVectorNotAvailable());
          return new_velocities[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n]->get_velocity();
  }


  template <int dim, int spacedim>
  void
  PartVectors<dim, spacedim>::set_velocity(
    const unsigned int                         part_n,
    const double                               time,
    LinearAlgebra::distributed::Vector<double> velocity)
  {
    AssertIndexRange(part_n, parts.size());
    velocity.set_ghost_state(false);
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          Assert(false, ExcMessage("cannot set velocity at current time"));
        case TimeStep::Half:
          half_velocities.resize(
            std::max(std::size_t(part_n + 1), half_velocities.size()));
          half_velocities[part_n].swap(velocity);
          return;
        case TimeStep::New:
          new_velocities.resize(
            std::max(std::size_t(part_n + 1), new_velocities.size()));
          new_velocities[part_n].swap(velocity);
          return;
      }

    Assert(false, ExcFDLInternalError());
  }



  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  PartVectors<dim, spacedim>::get_force(const unsigned int part_n,
                                        const double       time) const
  {
    AssertIndexRange(part_n, parts.size());
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          Assert(part_n < current_forces.size(), ExcVectorNotAvailable());
          return current_forces[part_n];
        case TimeStep::Half:
          Assert(part_n < half_forces.size(), ExcVectorNotAvailable());
          return half_forces[part_n];
        case TimeStep::New:
          Assert(part_n < new_forces.size(), ExcVectorNotAvailable());
          return new_forces[part_n];
      }

    Assert(false, ExcFDLInternalError());
    return parts[part_n]->get_position();
  }



  template <int dim, int spacedim>
  void
  PartVectors<dim, spacedim>::set_force(
    const unsigned int                         part_n,
    const double                               time,
    LinearAlgebra::distributed::Vector<double> force)
  {
    AssertIndexRange(part_n, parts.size());
    force.set_ghost_state(false);
    switch (get_time_step(time))
      {
        case TimeStep::Current:
          current_forces.resize(
            std::max(std::size_t(part_n + 1), current_forces.size()));
          current_forces[part_n].swap(force);
          return;
        case TimeStep::Half:
          half_forces.resize(
            std::max(std::size_t(part_n + 1), half_forces.size()));
          half_forces[part_n].swap(force);
          return;
        case TimeStep::New:
          new_forces.resize(
            std::max(std::size_t(part_n + 1), new_forces.size()));
          new_forces[part_n].swap(force);
          return;
      }

    Assert(false, ExcFDLInternalError());
  }

  template class PartVectors<NDIM - 1, NDIM>;
  template class PartVectors<NDIM, NDIM>;
} // namespace fdl
