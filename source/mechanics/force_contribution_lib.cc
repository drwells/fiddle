#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <algorithm>

namespace fdl
{
  using namespace dealii;

  namespace
  {
    template <int dim, int spacedim>
    LinearAlgebra::distributed::Vector<double>
    do_interpolation(const DoFHandler<dim, spacedim> &dof_handler,
                     const Mapping<dim, spacedim>    &mapping,
                     const Function<spacedim>        &reference_position)
    {
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      LinearAlgebra::distributed::Vector<double> result(
        dof_handler.locally_owned_dofs(),
        locally_relevant_dofs,
        dof_handler.get_triangulation().get_communicator());

      VectorTools::interpolate(mapping,
                               dof_handler,
                               reference_position,
                               result);

      return result;
    }

    // set up either material or boundary ids
    template <typename id_type>
    std::vector<id_type>
    setup_ids(const std::vector<id_type> &ids, const id_type invalid_id)
    {
      std::vector<id_type> result = ids;
      // If the user doesn't want any material/boundary ids, let them do it.
      // This utility function is only for the explicit material/boundary id
      // case.
      if (result.size() == 0)
        result.push_back(invalid_id);

      // permit duplicates in the input array
      std::sort(result.begin(), result.end());
      result.erase(std::unique(result.begin(), result.end()), result.end());

      return result;
    }
  } // namespace

  //
  // SpringForceBase
  //

  template <int dim, int spacedim, typename Number>
  template <int q_dim>
  SpringForceBase<dim, spacedim, Number>::SpringForceBase(
    const Quadrature<q_dim>                          &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : ForceContribution<dim, spacedim, double>(quad)
    , spring_constant(spring_constant)
    , dof_handler(&dof_handler)
    , reference_position(reference_position)
  {
    this->reference_position.update_ghost_values();
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForceBase<dim, spacedim, Number>::set_reference_position(
    const LinearAlgebra::distributed::Vector<double> &reference_position)
  {
    this->reference_position = reference_position;
    this->reference_position.update_ghost_values();
  }

  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  SpringForceBase<dim, spacedim, Number>::get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_nothing;
  }

  template <int dim, int spacedim, typename Number>
  UpdateFlags
  SpringForceBase<dim, spacedim, Number>::get_update_flags() const
  {
    return UpdateFlags::update_values;
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForceBase<dim, spacedim, Number>::setup_force(
    const double /*time*/,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> & /*velocity*/)
  {
    current_position = &position;
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForceBase<dim, spacedim, Number>::finish_force(const double /*time*/)
  {
    current_position = nullptr;
  }

  //
  // SpringForce
  //

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                            &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
  {}

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                            &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const std::vector<types::material_id>            &material_ids,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
    , material_ids(setup_ids(material_ids, numbers::invalid_material_id))
  {}

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                 &quad,
    const double                           spring_constant,
    const DoFHandler<dim, spacedim>       &dof_handler,
    const Mapping<dim, spacedim>          &mapping,
    const std::vector<types::material_id> &material_ids,
    const Function<spacedim>              &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
    , material_ids(setup_ids(material_ids, numbers::invalid_material_id))
  {}

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>           &quad,
    const double                     spring_constant,
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim>    &mapping,
    const Function<spacedim>        &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
  {}

  template <int dim, int spacedim, typename Number>
  void
  SpringForce<dim, spacedim, Number>::compute_volume_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim>                              &m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    ArrayView<Tensor<1, spacedim, Number>> &forces) const
  {
    if (this->material_ids.size() > 0 &&
        !std::binary_search(this->material_ids.begin(),
                            this->material_ids.end(),
                            cell->material_id()))
      {
        // the user specified a subset of material ids and we currently don't
        // match - fill with zeros
        for (auto &force : forces)
          force = 0.0;
      }
    else
      {
        const FEValuesBase<dim, spacedim> &fe_values = m_values.get_fe_values();

        const auto dof_cell =
          typename DoFHandler<dim, spacedim>::active_cell_iterator(
            &this->dof_handler->get_triangulation(),
            cell->level(),
            cell->index(),
            &*this->dof_handler);

        this->scratch_cell_dofs.resize(fe_values.dofs_per_cell);
        dof_cell->get_dof_indices(this->scratch_cell_dofs);
        this->scratch_dof_values.resize(fe_values.dofs_per_cell);
        this->scratch_qp_values.resize(fe_values.n_quadrature_points);

        auto &extractor = fe_values[FEValuesExtractors::Vector(0)];
        for (unsigned int i = 0; i < this->scratch_cell_dofs.size(); ++i)
          this->scratch_dof_values[i] =
            this->spring_constant *
            (this->reference_position[this->scratch_cell_dofs[i]] -
             (*this->current_position)[this->scratch_cell_dofs[i]]);
        extractor.get_function_values_from_local_dof_values(
          this->scratch_dof_values, this->scratch_qp_values);
        std::copy(this->scratch_qp_values.begin(),
                  this->scratch_qp_values.end(),
                  forces.begin());
      }
  }

  template <int dim, int spacedim, typename Number>
  bool
  SpringForce<dim, spacedim, Number>::is_volume_force() const
  {
    return true;
  }

  //
  // BoundarySpringForce
  //
  template <int dim, int spacedim, typename Number>
  BoundarySpringForce<dim, spacedim, Number>::BoundarySpringForce(
    const Quadrature<dim - 1>                        &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
  {}

  template <int dim, int spacedim, typename Number>
  BoundarySpringForce<dim, spacedim, Number>::BoundarySpringForce(
    const Quadrature<dim - 1>                        &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const std::vector<types::boundary_id>            &boundary_ids,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
    , boundary_ids(setup_ids(boundary_ids, numbers::invalid_boundary_id))
  {}

  template <int dim, int spacedim, typename Number>
  BoundarySpringForce<dim, spacedim, Number>::BoundarySpringForce(
    const Quadrature<dim - 1>             &quad,
    const double                           spring_constant,
    const DoFHandler<dim, spacedim>       &dof_handler,
    const Mapping<dim, spacedim>          &mapping,
    const std::vector<types::boundary_id> &boundary_ids,
    const Function<spacedim>              &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
    , boundary_ids(setup_ids(boundary_ids, numbers::invalid_boundary_id))
  {}

  template <int dim, int spacedim, typename Number>
  BoundarySpringForce<dim, spacedim, Number>::BoundarySpringForce(
    const Quadrature<dim - 1>       &quad,
    const double                     spring_constant,
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim>    &mapping,
    const Function<spacedim>        &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
  {}

  template <int dim, int spacedim, typename Number>
  bool
  BoundarySpringForce<dim, spacedim, Number>::is_boundary_force() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  BoundarySpringForce<dim, spacedim, Number>::compute_boundary_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim>                              &m_values,
    const typename Triangulation<dim, spacedim>::active_face_iterator &face,
    ArrayView<Tensor<1, spacedim, Number>> &forces) const
  {
    if (this->boundary_ids.size() > 0 &&
        !std::binary_search(this->boundary_ids.begin(),
                            this->boundary_ids.end(),
                            face->boundary_id()))
      {
        // the user specified a subset of boundary ids and we currently don't
        // match - fill with zeros
        for (auto &force : forces)
          force = 0.0;
      }
    else
      {
        const FEValuesBase<dim, spacedim> &fe_values = m_values.get_fe_values();
        const auto                         cell      = fe_values.get_cell();
        const auto                         dof_cell =
          typename DoFHandler<dim, spacedim>::active_cell_iterator(
            &this->dof_handler->get_triangulation(),
            cell->level(),
            cell->index(),
            &*this->dof_handler);

        this->scratch_cell_dofs.resize(fe_values.dofs_per_cell);
        dof_cell->get_dof_indices(this->scratch_cell_dofs);
        this->scratch_dof_values.resize(fe_values.dofs_per_cell);
        this->scratch_qp_values.resize(fe_values.n_quadrature_points);

        auto &extractor = fe_values[FEValuesExtractors::Vector(0)];
        for (unsigned int i = 0; i < this->scratch_cell_dofs.size(); ++i)
          this->scratch_dof_values[i] =
            this->spring_constant *
            (this->reference_position[this->scratch_cell_dofs[i]] -
             (*this->current_position)[this->scratch_cell_dofs[i]]);
        extractor.get_function_values_from_local_dof_values(
          this->scratch_dof_values, this->scratch_qp_values);
        std::copy(this->scratch_qp_values.begin(),
                  this->scratch_qp_values.end(),
                  forces.begin());
      }
  }

  //
  // DampingForce
  //
  template <int dim, int spacedim, typename Number>
  DampingForce<dim, spacedim, Number>::DampingForce(
    const Quadrature<dim> &quad,
    const double           damping_constant)
    : ForceContribution<dim, spacedim, double>(quad)
    , damping_constant(damping_constant)
  {}

  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  DampingForce<dim, spacedim, Number>::get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_velocity_values;
  }

  template <int dim, int spacedim, typename Number>
  bool
  DampingForce<dim, spacedim, Number>::is_volume_force() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  DampingForce<dim, spacedim, Number>::compute_volume_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim> &m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      & /*cell*/,
    ArrayView<Tensor<1, spacedim, Number>> &forces) const
  {
    std::copy(m_values.get_velocity_values().begin(),
              m_values.get_velocity_values().end(),
              forces.begin());
    for (auto &force : forces)
      force *= -damping_constant;
  }

  //
  // OrthogonalKelvinVoigtForce
  //
  template <int dim, int spacedim, typename Number>
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const double                                      damping_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
    , damping_constant(damping_constant)
  {}

  template <int dim, int spacedim, typename Number>
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>                        &quad,
      const double                                      spring_constant,
      const double                                      damping_constant,
      const DoFHandler<dim, spacedim>                  &dof_handler,
      const std::vector<types::boundary_id>            &boundary_ids,
      const LinearAlgebra::distributed::Vector<double> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(quad,
                                             spring_constant,
                                             dof_handler,
                                             reference_position)
    , damping_constant(damping_constant)
    , boundary_ids(setup_ids(boundary_ids, numbers::invalid_boundary_id))
  {}

  template <int dim, int spacedim, typename Number>
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::
    OrthogonalSpringDashpotForce(
      const Quadrature<dim - 1>             &quad,
      const double                           spring_constant,
      const double                           damping_constant,
      const DoFHandler<dim, spacedim>       &dof_handler,
      const Mapping<dim, spacedim>          &mapping,
      const std::vector<types::boundary_id> &boundary_ids,
      const Function<spacedim>              &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
    , damping_constant(damping_constant)
    , boundary_ids(setup_ids(boundary_ids, numbers::invalid_boundary_id))
  {}

  template <int dim, int spacedim, typename Number>
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::
    OrthogonalSpringDashpotForce(const Quadrature<dim - 1> &quad,
                                 const double               spring_constant,
                                 const double               damping_constant,
                                 const DoFHandler<dim, spacedim> &dof_handler,
                                 const Mapping<dim, spacedim>    &mapping,
                                 const Function<spacedim> &reference_position)
    : SpringForceBase<dim, spacedim, Number>(
        quad,
        spring_constant,
        dof_handler,
        do_interpolation(dof_handler, mapping, reference_position))
    , damping_constant(damping_constant)
  {}

  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::
    get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_velocity_values |
           MechanicsUpdateFlags::update_deformed_normal_vectors;
  }

  template <int dim, int spacedim, typename Number>
  bool
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::is_boundary_force() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  OrthogonalSpringDashpotForce<dim, spacedim, Number>::compute_boundary_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim>                              &m_values,
    const typename Triangulation<dim, spacedim>::active_face_iterator &face,
    ArrayView<Tensor<1, spacedim, Number>> &forces) const
  {
    if (this->boundary_ids.size() > 0 &&
        !std::binary_search(this->boundary_ids.begin(),
                            this->boundary_ids.end(),
                            face->boundary_id()))
      {
        // the user specified a subset of boundary ids and we currently don't
        // match - fill with zeros
        for (auto &force : forces)
          force = 0.0;
      }
    else
      {
        const FEValuesBase<dim, spacedim> &fe_values = m_values.get_fe_values();
        const auto                         cell      = fe_values.get_cell();
        const auto                         dof_cell =
          typename DoFHandler<dim, spacedim>::active_cell_iterator(
            &this->dof_handler->get_triangulation(),
            cell->level(),
            cell->index(),
            &*this->dof_handler);

        this->scratch_cell_dofs.resize(fe_values.dofs_per_cell);
        dof_cell->get_dof_indices(this->scratch_cell_dofs);
        this->scratch_dof_values.resize(fe_values.dofs_per_cell);
        this->scratch_qp_values.resize(fe_values.n_quadrature_points);

        auto &extractor = fe_values[FEValuesExtractors::Vector(0)];

        for (unsigned int i = 0; i < this->scratch_cell_dofs.size(); ++i)
          this->scratch_dof_values[i] =
            this->spring_constant *
            (this->reference_position[this->scratch_cell_dofs[i]] -
             (*this->current_position)[this->scratch_cell_dofs[i]]);

        extractor.get_function_values_from_local_dof_values(
          this->scratch_dof_values, this->scratch_qp_values);

        for (unsigned int i = 0; i < this->scratch_qp_values.size(); ++i)
          this->scratch_qp_values[i] =
            m_values.get_deformed_normal_vectors()[i] *
            (this->scratch_qp_values[i] -
             this->damping_constant * m_values.get_velocity_values()[i]) *
            m_values.get_deformed_normal_vectors()[i];

        std::copy(this->scratch_qp_values.begin(),
                  this->scratch_qp_values.end(),
                  forces.begin());
      }
  }

  template <int dim, int spacedim, typename Number>
  ModifiedNeoHookeanStress<dim, spacedim, Number>::ModifiedNeoHookeanStress(
    const Quadrature<dim> &                quad,
    const double                           shear_modulus,
    const std::vector<types::material_id> &material_ids)
    : ForceContribution<dim, spacedim, Number>(quad)
    , shear_modulus(shear_modulus)
    , material_ids(material_ids)
  {
    // permit duplicates in the input array
    std::sort(this->material_ids.begin(), this->material_ids.end());
    this->material_ids.erase(std::unique(this->material_ids.begin(),
                                         this->material_ids.end()),
                             this->material_ids.end());
  }

  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  ModifiedNeoHookeanStress<dim, spacedim, Number>::get_mechanics_update_flags()
    const
  {
    return MechanicsUpdateFlags::update_n23_det_FF |
           MechanicsUpdateFlags::update_FF |
           MechanicsUpdateFlags::update_FF_inv_T |
           MechanicsUpdateFlags::update_first_invariant;
  }

  template <int dim, int spacedim, typename Number>
  bool
  ModifiedNeoHookeanStress<dim, spacedim, Number>::is_stress() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  ModifiedNeoHookeanStress<dim, spacedim, Number>::compute_stress(
    const double /*time*/,
    const MechanicsValues<dim, spacedim> &                             m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    ArrayView<Tensor<2, spacedim, Number>> &stresses) const
  {
    if (this->material_ids.size() > 0 &&
        !std::binary_search(this->material_ids.begin(),
                            this->material_ids.end(),
                            cell->material_id()))
      {
        // the user specified a subset of material ids and we currently don't
        // match - fill with zeros
        for (auto &stress : stresses)
          stress = 0.0;
      }
    else
      {
        for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
          {
            stresses[qp_n] =
              shear_modulus * m_values.get_n23_det_FF()[qp_n] *
              (m_values.get_FF()[qp_n] - m_values.get_first_invariant()[qp_n] /
                                           3.0 * m_values.get_FF_inv_T()[qp_n]);
          }
      }
  }

  template <int dim, int spacedim, typename Number>
  ModifiedMooneyRivlinStress<dim, spacedim, Number>::ModifiedMooneyRivlinStress(
    const Quadrature<dim> &                quad,
    const double                           material_constant_1,
    const double                           material_constant_2,
    const std::vector<types::material_id> &material_ids)
    : ForceContribution<dim, spacedim, Number>(quad)
    , material_constant_1(material_constant_1)
    , material_constant_2(material_constant_2)
    , material_ids(material_ids)
  {
    // permit duplicates in the input array
    std::sort(this->material_ids.begin(), this->material_ids.end());
    this->material_ids.erase(std::unique(this->material_ids.begin(),
                                         this->material_ids.end()),
                             this->material_ids.end());
  }

  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  ModifiedMooneyRivlinStress<dim, spacedim, Number>::
    get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_n23_det_FF |
           MechanicsUpdateFlags::update_FF |
           MechanicsUpdateFlags::update_FF_inv_T |
           MechanicsUpdateFlags::update_first_invariant |
           MechanicsUpdateFlags::update_second_invariant |
           MechanicsUpdateFlags::update_right_cauchy_green;
  }

  template <int dim, int spacedim, typename Number>
  bool
  ModifiedMooneyRivlinStress<dim, spacedim, Number>::is_stress() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  ModifiedMooneyRivlinStress<dim, spacedim, Number>::compute_stress(
    const double /*time*/,
    const MechanicsValues<dim, spacedim> &                             m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    ArrayView<Tensor<2, spacedim, Number>> &stresses) const
  {
    if (this->material_ids.size() > 0 &&
        !std::binary_search(this->material_ids.begin(),
                            this->material_ids.end(),
                            cell->material_id()))
      {
        // the user specified a subset of material ids and we currently don't
        // match - fill with zeros
        for (auto &stress : stresses)
          stress = 0.0;
      }
    else
      {
        for (unsigned int qp_n = 0; qp_n < stresses.size(); ++qp_n)
          {
            const auto J_n23    = m_values.get_n23_det_FF()[qp_n];
            const auto FF       = m_values.get_FF()[qp_n];
            const auto FF_inv_T = m_values.get_FF_inv_T()[qp_n];
            const auto CC       = m_values.get_right_cauchy_green()[qp_n];
            const auto I1       = m_values.get_first_invariant()[qp_n];
            const auto I2       = m_values.get_second_invariant()[qp_n];

            stresses[qp_n] =
              2.0 * material_constant_1 * J_n23 * (FF - I1 / 3.0 * FF_inv_T) +
              2.0 * material_constant_2 * J_n23 * J_n23 *
                (I1 * FF - FF * CC - 2.0 * I2 / 3.0 * FF_inv_T);
          }
      }
  }


  template class SpringForceBase<NDIM - 1, NDIM, double>;
  template class SpringForceBase<NDIM, NDIM, double>;
  template class SpringForce<NDIM - 1, NDIM, double>;
  template class SpringForce<NDIM, NDIM, double>;
  template class BoundarySpringForce<NDIM - 1, NDIM, double>;
  template class BoundarySpringForce<NDIM, NDIM, double>;
  template class DampingForce<NDIM - 1, NDIM, double>;
  template class DampingForce<NDIM, NDIM, double>;
  template class OrthogonalSpringDashpotForce<NDIM - 1, NDIM, double>;
  template class OrthogonalSpringDashpotForce<NDIM, NDIM, double>;
  template class ModifiedNeoHookeanStress<NDIM, NDIM, double>;
  template class ModifiedMooneyRivlinStress<NDIM, NDIM, double>;
} // namespace fdl
