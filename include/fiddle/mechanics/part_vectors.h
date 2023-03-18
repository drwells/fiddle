#ifndef included_fiddle_part_vectors_h
#define included_fiddle_part_vectors_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/part.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Class managing various vectors related to a set of Parts.
   *
   * In IBAMR, different vectors are available at different points in the
   * time-step: for example, we initially only have the velocity at the
   * beginning of the timestep, but by the end of the timestep we will also have
   * it at the end. This class provides some basic checking to make sure we only
   * access vectors that exist at the requested time.
   */
  template <int dim, int spacedim = dim>
  class PartVectors
  {
  public:
    /**
     * Make the dimension available in function templates.
     */
    static constexpr unsigned int dimension = dim;

    /**
     * Make the space dimension available in function templates.
     */
    static constexpr unsigned int space_dimension = spacedim;

    /**
     * Constructor.
     */
    PartVectors(const std::vector<Part<dim, spacedim>> &parts);

    // Begin a new time step.
    void
    begin_time_step(const double current_time, const double new_time);

    // End a new time step.
    void
    end_time_step();

    // Get the correct temporary vector or (if time == current_time) the
    // vector from the Part object
    const LinearAlgebra::distributed::Vector<double> &
    get_position(const unsigned int part_n, const double time) const;

    // same, but for velocity
    const LinearAlgebra::distributed::Vector<double> &
    get_velocity(const unsigned int part_n, const double time) const;

    // same, but for force
    const LinearAlgebra::distributed::Vector<double> &
    get_force(const unsigned int part_n, const double time) const;

    // Set the position. Only valid for time != current_time.
    void
    set_position(const unsigned int                         part_n,
                 const double                               time,
                 LinearAlgebra::distributed::Vector<double> position);

    // Set the velocity. Only valid for time != current_time.
    void
    set_velocity(const unsigned int                         part_n,
                 const double                               time,
                 LinearAlgebra::distributed::Vector<double> velocity);

    // Set the force.
    void
    set_force(const unsigned int                         part_n,
              const double                               time,
              LinearAlgebra::distributed::Vector<double> force);

    // Get all new position vectors by moving them out of this object. Intended
    // to be called at the end of the time step.
    std::vector<LinearAlgebra::distributed::Vector<double>>
    get_all_new_positions();

    // Get all new velocity vectors by moving them out of this object. Intended
    // to be called at the end of the time step.
    std::vector<LinearAlgebra::distributed::Vector<double>>
    get_all_new_velocities();

    DeclExceptionMsg(ExcVectorNotAvailable,
                     "The requested vector is not available. This usually "
                     "occurs when this function is called with the wrong time "
                     "- e.g., requesting the position at the end of a time "
                     "step at the beginning of a time step.");

  protected:
    std::vector<SmartPointer<const Part<dim, spacedim>>> parts;

    enum class TimeStep
    {
      Current,
      Half,
      New
    };

    TimeStep
    get_time_step(const double time) const;

    double current_time;
    double half_time;
    double new_time;

    std::vector<LinearAlgebra::distributed::Vector<double>> current_forces;
    std::vector<LinearAlgebra::distributed::Vector<double>> half_forces;
    std::vector<LinearAlgebra::distributed::Vector<double>> new_forces;

    std::vector<LinearAlgebra::distributed::Vector<double>> half_positions;
    std::vector<LinearAlgebra::distributed::Vector<double>> new_positions;

    std::vector<LinearAlgebra::distributed::Vector<double>> half_velocities;
    std::vector<LinearAlgebra::distributed::Vector<double>> new_velocities;
  };

  // ----------------------------- inline functions ----------------------------

} // namespace fdl

#endif
