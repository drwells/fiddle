#ifndef included_fiddle_mechanics_part_h
#define included_fiddle_mechanics_part_h

#include <deal.II/base/bounding_box.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <fiddle/mechanics/mechanics_values.h>
#include <fiddle/mechanics/force_contribution.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Class encapsulating a single structure - essentially a wrapper that stores
   * the current position and velocity and can also compute the interior force
   * density.
   */
  template <int dim, int spacedim = dim>
  class Part
  {
  public:
    /**
     * Constructor.
     */
    Part(const Triangulation<dim, spacedim> &tria,
         const FiniteElement<dim, spacedim> &fe,
         std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                                   force_contributions = {},
         const Function<spacedim> &initial_position =
           Functions::IdentityFunction<spacedim>(),
         const Function<spacedim> &initial_velocity =
           Functions::ZeroFunction<spacedim>(spacedim));

    /**
     * Get a constant reference to the Triangulation.
     */
    const Triangulation<dim, spacedim> &
    get_triangulation() const
    {
      Assert(tria, ExcFDLInternalError());
      return *tria;
    }

    /**
     * Get a constant reference to the DoFHandler used for the position,
     * velocity, and force.
     */
    const DoFHandler<dim, spacedim> &
    get_dof_handler() const
    {
      return dof_handler;
    }

    /**
     * Get the shared vector partitioner for the position, velocity, and force.
     * Useful if users want to set up their own vectors and re-use the parallel
     * data layout for these finite element spaces.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner>
    get_partitioner() const
    {
      return partitioner;
    }

    /**
     * Get the current position of the structure.
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_position() const;

    /**
     * Set the current position by copying.
     */
    void
    set_position(const LinearAlgebra::distributed::Vector<double> &X);

    /**
     * Set the current position from a temporary.
     */
    void
    set_position(LinearAlgebra::distributed::Vector<double> &&X);

    /**
     * Get the current velocity of the structure.
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const;

    /**
     * Set the current velocity by copying.
     */
    void
    set_velocity(const LinearAlgebra::distributed::Vector<double> &X);

    /**
     * Set the current velocity from a temporary.
     */
    void
    set_velocity(LinearAlgebra::distributed::Vector<double> &&X);

  protected:
    /**
     * Triangulation of the part.
     */
    SmartPointer<const Triangulation<dim, spacedim>> tria;

    /**
     * Finite element for the position, velocity and force. Since velocity is
     * the time derivative of position we need to use the same FE for both
     * spaces. Similarly, to maintain adjointness between force spreading and
     * velocity interpolation, we need to use the same space for force and
     * velocity.
     */
    SmartPointer<const FiniteElement<dim, spacedim>> fe;

    /**
     * DoFHandler for the position, velocity, and force.
     */
    DoFHandler<dim, spacedim> dof_handler;

    /**
     * Partitioner for the position, velocity, and force vectors.
     */
    std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

    // Position.
    LinearAlgebra::distributed::Vector<double> position;

    // Velocity.
    LinearAlgebra::distributed::Vector<double> velocity;

    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
      force_contributions;
  };

  // Constructor

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    const Triangulation<dim, spacedim> &tria,
    const FiniteElement<dim, spacedim> &fe,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : tria(&tria)
    , fe(&fe)
    , dof_handler(tria)
  {
    Assert(fe.n_components() == spacedim,
           ExcMessage("The finite element should have spacedim components "
                      "since it will represent the position, velocity and "
                      "force of the part."));
    dof_handler.distribute_dofs(*this->fe);
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      locally_relevant_dofs,
      tria.get_communicator());
    position.reinit(partitioner);
    velocity.reinit(partitioner);

    VectorTools::interpolate(dof_handler, initial_position, position);
    VectorTools::interpolate(dof_handler, initial_velocity, velocity);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  // Functions for getting and setting state vectors

  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  Part<dim, spacedim>::get_position() const
  {
    return position;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_position(
    const LinearAlgebra::distributed::Vector<double> &pos)
  {
    // TODO loosen this check slightly or implement Partitioner::operator==
    Assert(pos.get_partitioner() == partitioner,
           ExcMessage("The partitioners must be equal"));
    position = pos;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_position(
    LinearAlgebra::distributed::Vector<double> &&pos)
  {
    // TODO loosen this check slightly or implement Partitioner::operator==
    Assert(pos.get_partitioner() == partitioner,
           ExcMessage("The partitioners must be equal"));
    position.swap(pos);
  }

  template <int dim, int spacedim>
  const LinearAlgebra::distributed::Vector<double> &
  Part<dim, spacedim>::get_velocity() const
  {
    return velocity;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_velocity(
    const LinearAlgebra::distributed::Vector<double> &vel)
  {
    // TODO loosen this check slightly or implement Partitioner::operator==
    Assert(vel.get_partitioner() == partitioner,
           ExcMessage("The partitioners must be equal"));
    velocity = vel;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_velocity(
    LinearAlgebra::distributed::Vector<double> &&vel)
  {
    // TODO loosen this check slightly or implement Partitioner::operator==
    Assert(vel.get_partitioner() == partitioner,
           ExcMessage("The partitioners must be equal"));
    velocity.swap(vel);
  }
} // namespace fdl

#endif
