#include <fiddle/mechanics/part.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

namespace fdl
{
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
    dof_handler->distribute_dofs(*this->fe);
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                            locally_relevant_dofs);

    partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler->locally_owned_dofs(),
      locally_relevant_dofs,
      tria.get_communicator());
    position.reinit(partitioner);
    velocity.reinit(partitioner);

    VectorTools::interpolate(*dof_handler, initial_position, position);
    // The initial velocity is probably zero:
    if (dynamic_cast<const Functions::ZeroFunction<dim> *>(&initial_velocity))
      velocity = 0.0;
    else
      VectorTools::interpolate(*dof_handler, initial_velocity, velocity);

    position.update_ghost_values();
    velocity.update_ghost_values();
  }

  template class Part<NDIM - 1, NDIM>;
  template class Part<NDIM, NDIM>;
} // namespace fdl
