#ifndef included_fiddle_mechanics_part_h
#define included_fiddle_mechanics_part_h

#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <mpi.h>

#include <memory>
#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Class encapsulating a single structure - essentially a wrapper that stores
   * the current position and velocity and can also compute the interior force
   * density.
   *
   * The primary intent of this class it to encapsulate the state of the finite
   * element discretization in a single place. This class is responsible for
   * managing the current position and velocity of a structure, as well as all
   * the finite element book-keeping (e.g., the mass operator).
   *
   * @todo In the future we should add an API that allows users to merge in
   * their own constraints to the position, force, or displacement systems. This
   * class also needs to learn how to set up hanging node constraints. This
   * might not be trivial - if we constrain the position space then that implies
   * constraints on the velocity space. This might also raise adjointness
   * concerns.
   */
  template <int dim, int spacedim = dim>
  class Part : public Subscriptor
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
     * Constructor, which uses an externally managed DoFHandler.
     */
    Part(std::shared_ptr<DoFHandler<dim, spacedim>> dof_handler,
         std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                                   force_contributions = {},
         const Function<spacedim> &initial_position =
           Functions::IdentityFunction<spacedim>(),
         const Function<spacedim> &initial_velocity =
           Functions::ZeroFunction<spacedim>(spacedim));

    /**
     * Save the current state of the object to an archive.
     *
     * @note at the present time no information from the force contributions is
     * saved. This may change in the future.
     */
    void
    save(boost::archive::binary_oarchive &archive,
         const unsigned int               version) const;

    /**
     * Load a previously saved state of the object from an archive.
     *
     * This function only makes sense if the Part has been set up in the same
     * way as the one in the archive - in particular, they must use the same
     * parallel data distribution and finite element spaces.
     *
     * @note at the present time no information from the force contributions is
     * loaded. This may change in the future.
     */
    void
    load(boost::archive::binary_iarchive &archive, const unsigned int version);

    /**
     * Move constructor.
     */
    Part(Part<dim, spacedim> &&part) = default;

    /**
     * Get a constant reference to the Triangulation.
     */
    const Triangulation<dim, spacedim> &
    get_triangulation() const;

    /**
     * Get a copy of the communicator.
     */
    MPI_Comm
    get_communicator() const;

    /**
     * Get pointers to the force contributions.
     */
    std::vector<ForceContribution<dim, spacedim> *>
    get_force_contributions() const;

    /**
     * Add another force contribution.
     *
     * Typically, force contributions should be set by the constructor, but it
     * is sometimes necessary to add additional forces later on.
     */
    void
    add_force_contribution(std::unique_ptr<ForceContribution<dim, spacedim>> force);

    /**
     * Get a constant reference to the DoFHandler used for the position,
     * velocity, and force.
     */
    const DoFHandler<dim, spacedim> &
    get_dof_handler() const;

    /**
     * Get the shared vector partitioner for the position, velocity, and force.
     * Useful if users want to set up their own vectors and re-use the parallel
     * data layout for these finite element spaces.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner>
    get_partitioner() const;

    /**
     * Return a reference to the quadrature used to set up the mass operator.
     */
    const Quadrature<dim> &
    get_quadrature() const;

    /**
     * Return a reference to the mapping used to set up the mass operator.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const;

    /**
     * Get the mass operator.
     */
    const MatrixFreeOperators::Base<dim> &
    get_mass_operator() const;

    /**
     * Get the preconditioner associated with the mass operator.
     */
    const PreconditionJacobi<MatrixFreeOperators::Base<dim>> &
    get_mass_preconditioner() const;

    /**
     * Get the current position of the structure.
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_position() const;

    /**
     * Set the current position by copying.
     */
    void
    set_position(const LinearAlgebra::distributed::Vector<double> &position);

    /**
     * Set the current position from a temporary.
     */
    void
    set_position(LinearAlgebra::distributed::Vector<double> &&position);

    /**
     * Get the current velocity of the structure.
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const;

    /**
     * Set the current velocity by copying.
     */
    void
    set_velocity(const LinearAlgebra::distributed::Vector<double> &position);

    /**
     * Set the current velocity from a temporary.
     */
    void
    set_velocity(LinearAlgebra::distributed::Vector<double> &&position);

  protected:
    /**
     * Actual archive function. Separate for now from load and save.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version);

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
     * DoFHandler for the position, velocity, and force. May be shared by
     * external applications.
     */
    std::shared_ptr<DoFHandler<dim, spacedim>> dof_handler;

    /**
     * Constraints on the position, velocity, and force. Presently empty.
     */
    AffineConstraints<double> constraints;

    /**
     * Partitioner for the position, velocity, and force vectors.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

    // Quadrature used for the position, velocity, and force.
    Quadrature<dim> quadrature;

    // Mapping used for the position, velocity, and force.
    std::unique_ptr<Mapping<dim, spacedim>> mapping;

    // MatrixFree object.
    std::shared_ptr<MatrixFree<dim, double>> matrix_free;

    // Mass operator. Used for L2 projections.
    std::unique_ptr<MatrixFreeOperators::Base<dim>> mass_operator;

    // Preconditioner.
    PreconditionJacobi<MatrixFreeOperators::Base<dim>> mass_preconditioner;

    // Position.
    LinearAlgebra::distributed::Vector<double> position;

    // Velocity.
    LinearAlgebra::distributed::Vector<double> velocity;

    // All the functions that compute part of the force.
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
      force_contributions;
  };

  // ----------------------------- inline functions ----------------------------

  // Functions for getting basic objects owned by the Part

  template <int dim, int spacedim>
  const Triangulation<dim, spacedim> &
  Part<dim, spacedim>::get_triangulation() const
  {
    Assert(tria, ExcFDLInternalError());
    return *tria;
  }

  template <int dim, int spacedim>
  MPI_Comm
  Part<dim, spacedim>::get_communicator() const
  {
    Assert(tria, ExcFDLInternalError());
    return tria->get_communicator();
  }

  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  Part<dim, spacedim>::get_dof_handler() const
  {
    return *dof_handler;
  }

  template <int dim, int spacedim>
  std::shared_ptr<const Utilities::MPI::Partitioner>
  Part<dim, spacedim>::get_partitioner() const
  {
    return partitioner;
  }

  template <int dim, int spacedim>
  const Quadrature<dim> &
  Part<dim, spacedim>::get_quadrature() const
  {
    return quadrature;
  }

  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  Part<dim, spacedim>::get_mapping() const
  {
    return *mapping;
  }

  template <int dim, int spacedim>
  const MatrixFreeOperators::Base<dim> &
  Part<dim, spacedim>::get_mass_operator() const
  {
    return *mass_operator;
  }

  template <int dim, int spacedim>
  const PreconditionJacobi<MatrixFreeOperators::Base<dim>> &
  Part<dim, spacedim>::get_mass_preconditioner() const
  {
    return mass_preconditioner;
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
    Assert(partitioner->is_compatible(*pos.get_partitioner()),
           ExcMessage("The partitioners must be compatible"));
    position = pos;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_position(
    LinearAlgebra::distributed::Vector<double> &&pos)
  {
    Assert(partitioner->is_compatible(*pos.get_partitioner()),
           ExcMessage("The partitioners must be compatible"));
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
    Assert(partitioner->is_compatible(*vel.get_partitioner()),
           ExcMessage("The partitioners must be compatible"));
    velocity = vel;
  }

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::set_velocity(
    LinearAlgebra::distributed::Vector<double> &&vel)
  {
    Assert(partitioner->is_compatible(*vel.get_partitioner()),
           ExcMessage("The partitioners must be compatible"));
    velocity.swap(vel);
  }
} // namespace fdl

#endif
