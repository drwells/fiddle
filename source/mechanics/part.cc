#include <fiddle/mechanics/part.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/reference_cell.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

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
    const Triangulation<dim, spacedim> &tria,
    const FiniteElement<dim, spacedim> &fe,
    std::vector<std::unique_ptr<ForceContribution<dim, spacedim>>>
                              force_contributions,
    const Function<spacedim> &initial_position,
    const Function<spacedim> &initial_velocity)
    : tria(&tria)
    , fe(&fe)
    , dof_handler(std::make_unique<DoFHandler<dim, spacedim>>(tria))
    , force_contributions(std::move(force_contributions))
  {
    Assert(fe.n_components() == spacedim,
           ExcMessage("The finite element should have spacedim components "
                      "since it will represent the position, velocity and "
                      "force of the part."));
    // Set up DoFs and finite element fields:
    dof_handler->distribute_dofs(*this->fe);
    constraints.close();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                            locally_relevant_dofs);

    partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler->locally_owned_dofs(),
      locally_relevant_dofs,
      tria.get_communicator());
    position.reinit(partitioner);
    velocity.reinit(partitioner);

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

    // Set up matrix free components:
    if (dim == spacedim)
      {
        using namespace MatrixFreeOperators;
        switch (fe.tensor_degree())
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

        matrix_free = std::make_shared<MatrixFree<dim, double>>();
        internal::reinit_matrix_free(
          *mapping, *dof_handler, constraints, quadrature, *matrix_free);

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



  template <int dim, int spacedim>
  std::vector<const ForceContribution<dim, spacedim> *>
  Part<dim, spacedim>::get_stress_contributions() const
  {
    std::vector<const ForceContribution<dim, spacedim> *> stresses;

    for (const auto &force_contribution : force_contributions)
      {
        if (force_contribution->is_stress())
          stresses.push_back(force_contribution.get());
      }

    return stresses;
  }



  template class Part<NDIM - 1, NDIM>;
  template class Part<NDIM, NDIM>;
} // namespace fdl
