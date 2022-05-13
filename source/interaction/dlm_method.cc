#include <fiddle/interaction/dlm_method.h>

namespace fdl
{
  template <int dim, int spacedim>
  DLMForce<dim, spacedim>::DLMForce(
    const Quadrature<dim>              &quad,
    const double                        spring_constant,
    const double                        damping_constant,
    const DoFHandler<dim, spacedim>    &dof_handler,
    const DLMMethodBase<dim, spacedim> &dlm)
    : ForceContribution<dim, spacedim>(quad)
    , spring_constant(spring_constant)
    , damping_constant(damping_constant)
    , dof_handler(&dof_handler)
    , dlm(&dlm)
    , reference_position(this->dlm->get_current_mechanics_position())
    , reference_velocity(this->dlm->get_current_mechanics_velocity())
  {}

  template <int dim, int spacedim>
  MechanicsUpdateFlags
  DLMForce<dim, spacedim>::get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_nothing;
  }

  template <int dim, int spacedim>
  UpdateFlags
  DLMForce<dim, spacedim>::get_update_flags() const
  {
    return UpdateFlags::update_values;
  }

  template <int dim, int spacedim>
  bool
  DLMForce<dim, spacedim>::is_volume_force() const
  {
    return true;
  }

  template <int dim, int spacedim>
  void
  DLMForce<dim, spacedim>::setup_force(
    const double                                      time,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
  {
    current_position = &position;
    current_velocity = &velocity;
    dlm->get_mechanics_position(time, reference_position);
    dlm->get_mechanics_velocity(time, reference_velocity);
  }

  template <int dim, int spacedim>
  void
  DLMForce<dim, spacedim>::finish_force(const double /*time*/)
  {
    this->current_position = nullptr;
    this->current_velocity = nullptr;
  }

  template <int dim, int spacedim>
  void
  DLMForce<dim, spacedim>::compute_volume_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim>                              &m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    ArrayView<Tensor<1, spacedim>> &forces) const
  {
     const FEValuesBase<dim, spacedim> &fe_values = m_values.get_fe_values();

     const auto dof_cell =
       typename DoFHandler<dim, spacedim>::active_cell_iterator(
         &dof_handler->get_triangulation(),
         cell->level(),
         cell->index(),
         &*dof_handler);

     scratch_cell_dofs.resize(fe_values.dofs_per_cell);
     dof_cell->get_dof_indices(scratch_cell_dofs);
     this->scratch_dof_values.resize(fe_values.dofs_per_cell);
     this->scratch_qp_values.resize(fe_values.n_quadrature_points);

     auto &extractor = fe_values[FEValuesExtractors::Vector(0)];

     for (unsigned int i = 0; i < forces.size(); ++i)
       forces[i] = Tensor<1, dim>();
     if (spring_constant != 0.0)
       {
         for (unsigned int i = 0; i < scratch_cell_dofs.size(); ++i)
           scratch_dof_values[i] =
             spring_constant *
             (reference_position[scratch_cell_dofs[i]] -
              (*current_position)[scratch_cell_dofs[i]]);
         extractor.get_function_values_from_local_dof_values(
           scratch_dof_values, scratch_qp_values);
         std::copy(scratch_qp_values.begin(),
                   scratch_qp_values.end(),
                   forces.begin());
       }
     if (damping_constant != 0.0)
       {
         for (unsigned int i = 0; i < scratch_cell_dofs.size(); ++i)
           scratch_dof_values[i] =
             damping_constant *
             (reference_velocity[scratch_cell_dofs[i]] -
              (*current_velocity)[scratch_cell_dofs[i]]);
         extractor.get_function_values_from_local_dof_values(
           scratch_dof_values, scratch_qp_values);
         for (unsigned int i = 0; i < scratch_cell_dofs.size(); ++i)
           forces[i] += scratch_qp_values[i];
       }
  }


  // template class DLMForce<NDIM - 1, NDIM>;
  template class DLMForce<NDIM, NDIM>;
} // namespace fdl
