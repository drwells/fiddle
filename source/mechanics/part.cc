#include <fiddle/mechanics/part.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <boost/serialization/array_wrapper.hpp>

namespace fdl
{
  namespace internal
  {
    // matrix_free doesn't work with codim != 0 so we need a helper function
    template <int dim>
    void
    reinit_matrix_free(const Mapping<dim> &             mapping,
                       const DoFHandler<dim> &          dof_handler,
                       const AffineConstraints<double> &constraints,
                       const Quadrature<dim> &          quadrature,
                       MatrixFree<dim, double> &        matrix_free)
    {
      matrix_free.reinit(mapping, dof_handler, constraints, quadrature);
    }

    template <int dim>
    void
    reinit_matrix_free(const Mapping<dim - 1, dim> &,
                       const DoFHandler<dim - 1, dim> &,
                       const AffineConstraints<double> &,
                       const Quadrature<dim - 1> &,
                       MatrixFree<dim - 1, double> &)
    {
      // We shouldn't get here
      Assert(false, ExcFDLInternalError());
    }
  } // namespace internal

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    std::shared_ptr<DoFHandler<dim, spacedim>> dh,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : tria(&dh->get_triangulation())
    , fe(&dh->get_fe())
    , dof_handler(dh)
    , force_contributions(std::move(force_contributions))
  {
    // TODO - make the quadrature and mapping parameters so we can implement
    // nodal interaction
    const auto &reference_cells = this->tria->get_reference_cells();
    Assert(reference_cells.size() == 1, ExcFDLNotImplemented());
    mapping = reference_cells.front()
                .template get_default_linear_mapping<dim, spacedim>()
                .clone();
    quadrature =
      reference_cells.front().template get_gauss_type_quadrature<dim>(
        this->fe->tensor_degree() + 1);

    Assert(fe->n_components() == spacedim,
           ExcMessage("The finite element should have spacedim components "
                      "since it will represent the position, velocity and "
                      "force of the part."));
    // Set up DoFs and finite element fields:
    dof_handler->distribute_dofs(*this->fe);
    constraints.close();

    // A MatrixFree object sets up the partitioning on its own - use that to
    // avoid issues with p::s::T where there may not be artificial cells/
    //
    // TODO - understand this issue well enough to file a bug report
    matrix_free = std::make_shared<MatrixFree<dim, double>>();
    internal::reinit_matrix_free(
      *mapping, *dof_handler, constraints, quadrature, *matrix_free);
    if (dim == spacedim)
      {
        // no matrixfree outside codim 0
        partitioner = matrix_free->get_vector_partitioner();
      }
    else
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                                locally_relevant_dofs);
        partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler->locally_owned_dofs(),
          locally_relevant_dofs,
          tria->get_communicator());
      }

    position.reinit(partitioner);
    velocity.reinit(partitioner);

    // Set up matrix free components:
    if (dim == spacedim)
      {
        if (reference_cells.front() == ReferenceCells::get_hypercube<dim>())
          {
            using namespace MatrixFreeOperators;
            switch (fe->tensor_degree())
              {
                case 1:
                  mass_operator.reset(new MassOperator<dim, 1, 1 + 1, dim>());
                  break;
                case 2:
                  mass_operator.reset(new MassOperator<dim, 2, 2 + 1, dim>());
                  break;
                case 3:
                  mass_operator.reset(new MassOperator<dim, 3, 3 + 1, dim>());
                  break;
                case 4:
                  mass_operator.reset(new MassOperator<dim, 4, 4 + 1, dim>());
                  break;
                case 5:
                  mass_operator.reset(new MassOperator<dim, 5, 5 + 1, dim>());
                  break;
                default:
                  Assert(false, ExcFDLNotImplemented());
              }
          }
        else
          {
            using namespace MatrixFreeOperators;
            switch (fe->tensor_degree())
              {
                case 1:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                case 2:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                case 3:
                  mass_operator.reset(new MassOperator<dim, -1, 0, dim>());
                  break;
                default:
                  Assert(false, ExcFDLNotImplemented());
              }
          }
        mass_operator->initialize(matrix_free);
        mass_operator->compute_diagonal();
        mass_preconditioner.initialize(*mass_operator, 1.0);
      }

    // finally, FE fields:
    VectorTools::interpolate(*dof_handler, initial_position, position);
    // The initial velocity is probably zero:
    if (dynamic_cast<const Functions::ZeroFunction<dim> *>(&initial_velocity))
      velocity = 0.0;
    else
      VectorTools::interpolate(*dof_handler, initial_velocity, velocity);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  namespace
  {
    template <int dim, int spacedim>
    std::shared_ptr<DoFHandler<dim, spacedim>>
    setup_dof_handler(const Triangulation<dim, spacedim> &tria,
                      const FiniteElement<dim, spacedim> &fe)
    {
      auto dof_handler = std::make_shared<DoFHandler<dim, spacedim>>(tria);
      dof_handler->distribute_dofs(fe);
      return dof_handler;
    }
  } // namespace

  template <int dim, int spacedim>
  Part<dim, spacedim>::Part(
    const Triangulation<dim, spacedim> &tria,
    const FiniteElement<dim, spacedim> &fe,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : Part(setup_dof_handler(tria, fe),
           std::move(force_contributions),
           initial_position,
           initial_velocity)
  {}

  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::save(boost::archive::binary_oarchive &archive,
                            const unsigned int               version) const
  {
    const_cast<Part<dim, spacedim> *>(this)->serialize(archive, version);
  }


  template <int dim, int spacedim>
  void
  Part<dim, spacedim>::load(boost::archive::binary_iarchive &archive,
                            const unsigned int               version)
  {
    serialize(archive, version);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  template <int dim, int spacedim>
  template <class Archive>
  void
  Part<dim, spacedim>::serialize(Archive &ar, const unsigned int /*version*/)
  {
    boost::serialization::array_wrapper<double> position_wrapper(
      position.begin(), position.locally_owned_size());
    boost::serialization::array_wrapper<double> velocity_wrapper(
      velocity.begin(), velocity.locally_owned_size());

    ar &position_wrapper;
    ar &velocity_wrapper;
  }

  template <int dim, int spacedim>
  std::vector<ForceContribution<dim, spacedim> *>
  Part<dim, spacedim>::get_stress_contributions() const
  {
    std::vector<ForceContribution<dim, spacedim> *> stresses;

    for (auto &force_contribution : force_contributions)
      {
        if (force_contribution->is_stress())
          stresses.push_back(force_contribution.get());
      }

    return stresses;
  }

  template <int dim, int spacedim>
  std::vector<ForceContribution<dim, spacedim> *>
  Part<dim, spacedim>::get_volumetric_force_contributions() const
  {
    std::vector<ForceContribution<dim, spacedim> *> forces;

    for (auto &force_contribution : force_contributions)
      {
        if (force_contribution->is_volume_force())
          forces.push_back(force_contribution.get());
      }

    return forces;
  }


  template <int dim, int spacedim>
  std::vector<ForceContribution<dim, spacedim> *>
  Part<dim, spacedim>::get_boundary_force_contributions() const
  {
    std::vector<ForceContribution<dim, spacedim> *> forces;

    for (auto &force_contribution : force_contributions)
      {
        if (force_contribution->is_boundary_force())
          forces.push_back(force_contribution.get());
      }

    return forces;
  }



  template class Part<NDIM - 1, NDIM>;
  template class Part<NDIM, NDIM>;
} // namespace fdl
