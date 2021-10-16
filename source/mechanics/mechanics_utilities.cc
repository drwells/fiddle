#include <fiddle/mechanics/mechanics_utilities.h>
#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Compute the volumetric component of the PK1 stress and add it into the
   * given load vector.
   */
  template <int dim, int spacedim>
  void
  compute_volumetric_pk1_load_vector(
    const DoFHandler<dim, spacedim> &                      dof_handler,
    const Mapping<dim, spacedim> &                         mapping,
    const std::vector<ForceContribution<dim, spacedim> *> &stress_contributions,
    const double                                           time,
    const LinearAlgebra::distributed::Vector<double> &     current_position,
    const LinearAlgebra::distributed::Vector<double> &     current_velocity,
    LinearAlgebra::distributed::Vector<double> &           force_rhs)
  {
    Assert(dim == spacedim, ExcNotImplemented());

    if (stress_contributions.size() == 0)
      return;

    for (const auto *p : stress_contributions)
      {
        Assert(p, ExcMessage("stresses should not be nullptr"));
        Assert(p->is_stress(), ExcMessage("only valid for stresses"));
      }

    // Batch stresses by the quadrature rules they use
    std::vector<ForceContribution<dim, spacedim> *> remaining_stresses =
      stress_contributions;
    do
      {
        auto                   exemplar_stress = remaining_stresses.front();
        const Quadrature<dim> &exemplar_quadrature =
          exemplar_stress->get_cell_quadrature();
        const auto next_group_start =
          std::partition(remaining_stresses.begin(),
                         remaining_stresses.end(),
                         [&](const ForceContribution<dim, spacedim> *p) {
                           return p->get_cell_quadrature() ==
                                  exemplar_quadrature;
                         });

        std::vector<ForceContribution<dim, spacedim> *> current_stresses(
          remaining_stresses.begin(), next_group_start);
        remaining_stresses.erase(remaining_stresses.begin(), next_group_start);

        // Collect common flags:
        MechanicsUpdateFlags me_flags = MechanicsUpdateFlags::update_nothing;
        UpdateFlags          update_flags = UpdateFlags::update_default;
        for (const auto *stress : current_stresses)
          {
            me_flags |= stress->get_mechanics_update_flags();
            update_flags |= stress->get_update_flags();
          }
        update_flags |= compute_flag_dependencies(me_flags);
        // Add the stuff we need here too:
        update_flags |= update_gradients | update_JxW_values;
        const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();

        FEValues<dim, spacedim> fe_values(mapping,
                                          fe,
                                          exemplar_quadrature,
                                          update_flags);
        MechanicsValues<dim,
                        spacedim,
                        LinearAlgebra::distributed::Vector<double>>
          me_values(fe_values, current_position, current_velocity, me_flags);

        const unsigned int n_quadrature_points = exemplar_quadrature.size();
        std::vector<types::global_dof_index>     cell_dofs(fe.dofs_per_cell);
        std::vector<double>                      cell_rhs(fe.dofs_per_cell);
        std::vector<Tensor<2, spacedim, double>> one_stress(
          n_quadrature_points);
        std::vector<Tensor<2, spacedim, double>> accumulated_stresses(
          n_quadrature_points);
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(cell_dofs);
                fe_values.reinit(cell);
                me_values.reinit();
                std::fill(accumulated_stresses.begin(),
                          accumulated_stresses.end(),
                          Tensor<2, spacedim, double>());
                std::fill(cell_rhs.begin(), cell_rhs.end(), 0.0);
                auto &extractor = fe_values[FEValuesExtractors::Vector(0)];

                // Compute stresses at quadrature points
                for (const ForceContribution<dim, spacedim> *fc :
                     current_stresses)
                  {
                    std::fill(one_stress.begin(),
                              one_stress.end(),
                              Tensor<2, spacedim, double>());
                    auto view =
                      make_array_view(one_stress.begin(), one_stress.end());
                    fc->compute_stress(time, me_values, cell, view);
                    for (unsigned int qp_n = 0; qp_n < n_quadrature_points;
                         ++qp_n)
                      accumulated_stresses[qp_n] += one_stress[qp_n];
                  }

                // Assemble the RHS vector
                //
                // TODO - we could make this a lot faster by exploiting the fact
                // that we have primitive FEs most of the time
                for (unsigned int qp_n = 0; qp_n < n_quadrature_points; ++qp_n)
                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    // -PP : grad phi dx
                    cell_rhs[i] += -1. *
                                   scalar_product(accumulated_stresses[qp_n],
                                                  extractor.gradient(i, qp_n)) *
                                   fe_values.JxW(qp_n);

                force_rhs.add(cell_dofs, cell_rhs);
              }
          }
    } while (remaining_stresses.size() > 0);
  }

  template <int dim, int spacedim>
  void
  compute_volumetric_force_load_vector(
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim> &   mapping,
    const std::vector<ForceContribution<dim, spacedim> *>
      &          volume_force_contributions,
    const double time,
    const LinearAlgebra::distributed::Vector<double> &current_position,
    const LinearAlgebra::distributed::Vector<double> &current_velocity,
    LinearAlgebra::distributed::Vector<double> &      force_rhs)
  {
    Assert(dim == spacedim, ExcNotImplemented());

    if (volume_force_contributions.size() == 0)
      return;

    for (const auto *p : volume_force_contributions)
      {
        Assert(p, ExcMessage("volume forces should not be nullptr"));
        Assert(p->is_volume_force(),
               ExcMessage("only valid for volume forces"));
      }

    // Batch forces by the quadrature rules they use
    std::vector<ForceContribution<dim, spacedim> *> remaining_forces =
      volume_force_contributions;
    do
      {
        auto                   exemplar_force = remaining_forces.front();
        const Quadrature<dim> &exemplar_quadrature =
          exemplar_force->get_cell_quadrature();
        const auto next_group_start =
          std::partition(remaining_forces.begin(),
                         remaining_forces.end(),
                         [&](const ForceContribution<dim, spacedim> *p) {
                           return p->get_cell_quadrature() ==
                                  exemplar_quadrature;
                         });

        std::vector<ForceContribution<dim, spacedim> *> current_forces(
          remaining_forces.begin(), next_group_start);
        remaining_forces.erase(remaining_forces.begin(), next_group_start);

        // Collect common flags:
        MechanicsUpdateFlags me_flags = MechanicsUpdateFlags::update_nothing;
        UpdateFlags          update_flags = UpdateFlags::update_default;
        for (const auto *force : current_forces)
          {
            me_flags |= force->get_mechanics_update_flags();
            update_flags |= force->get_update_flags();
          }
        update_flags |= compute_flag_dependencies(me_flags);
        // Add the stuff we need here too:
        update_flags |= update_values | update_JxW_values;
        const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();

        FEValues<dim, spacedim> fe_values(mapping,
                                          fe,
                                          exemplar_quadrature,
                                          update_flags);
        MechanicsValues<dim,
                        spacedim,
                        LinearAlgebra::distributed::Vector<double>>
          me_values(fe_values, current_position, current_velocity, me_flags);

        const unsigned int n_quadrature_points = exemplar_quadrature.size();
        std::vector<types::global_dof_index>     cell_dofs(fe.dofs_per_cell);
        std::vector<double>                      cell_rhs(fe.dofs_per_cell);
        std::vector<Tensor<1, spacedim, double>> one_force(n_quadrature_points);
        std::vector<Tensor<1, spacedim, double>> accumulated_forces(
          n_quadrature_points);
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(cell_dofs);
                fe_values.reinit(cell);
                me_values.reinit();
                std::fill(accumulated_forces.begin(),
                          accumulated_forces.end(),
                          Tensor<1, spacedim, double>());
                std::fill(cell_rhs.begin(), cell_rhs.end(), 0.0);
                auto &extractor = fe_values[FEValuesExtractors::Vector(0)];

                // Compute forces at quadrature points
                for (const ForceContribution<dim, spacedim> *fc :
                     current_forces)
                  {
                    std::fill(one_force.begin(),
                              one_force.end(),
                              Tensor<1, spacedim, double>());
                    auto view =
                      make_array_view(one_force.begin(), one_force.end());
                    fc->compute_volume_force(time, me_values, cell, view);
                    for (unsigned int qp_n = 0; qp_n < n_quadrature_points;
                         ++qp_n)
                      accumulated_forces[qp_n] += one_force[qp_n];
                  }

                // Assemble the RHS vector
                //
                // TODO - we could make this a lot faster by exploiting the fact
                // that we have primitive FEs most of the time
                for (unsigned int qp_n = 0; qp_n < n_quadrature_points; ++qp_n)
                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    // F . phi dx
                    cell_rhs[i] += scalar_product(accumulated_forces[qp_n],
                                                  extractor.value(i, qp_n)) *
                                   fe_values.JxW(qp_n);

                force_rhs.add(cell_dofs, cell_rhs);
              }
          }
    } while (remaining_forces.size() > 0);
  }


  template <int dim, int spacedim>
  void
  compute_boundary_force_load_vector(
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim> &   mapping,
    const std::vector<ForceContribution<dim, spacedim> *>
      &          boundary_force_contributions,
    const double time,
    const LinearAlgebra::distributed::Vector<double> &current_position,
    const LinearAlgebra::distributed::Vector<double> &current_velocity,
    LinearAlgebra::distributed::Vector<double> &      force_rhs)
  {
    Assert(dim == spacedim, ExcNotImplemented());

    if (boundary_force_contributions.size() == 0)
      return;

    for (const auto *p : boundary_force_contributions)
      {
        Assert(p, ExcMessage("boundary forces should not be nullptr"));
        Assert(p->is_boundary_force(),
               ExcMessage("only valid for boundary forces"));
      }

    // Batch forces by the quadrature rules they use
    std::vector<ForceContribution<dim, spacedim> *> remaining_forces =
      boundary_force_contributions;
    do
      {
        auto                       exemplar_force = remaining_forces.front();
        const Quadrature<dim - 1> &exemplar_quadrature =
          exemplar_force->get_face_quadrature();
        const auto next_group_start =
          std::partition(remaining_forces.begin(),
                         remaining_forces.end(),
                         [&](const ForceContribution<dim, spacedim> *p) {
                           return p->get_face_quadrature() ==
                                  exemplar_quadrature;
                         });

        std::vector<ForceContribution<dim, spacedim> *> current_forces(
          remaining_forces.begin(), next_group_start);
        remaining_forces.erase(remaining_forces.begin(), next_group_start);

        // Collect common flags:
        MechanicsUpdateFlags me_flags = MechanicsUpdateFlags::update_nothing;
        UpdateFlags          update_flags = UpdateFlags::update_default;
        for (const auto *force : current_forces)
          {
            me_flags |= force->get_mechanics_update_flags();
            update_flags |= force->get_update_flags();
          }
        update_flags |= compute_flag_dependencies(me_flags);
        // Add the stuff we need here too:
        update_flags |= update_values | update_JxW_values;
        const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();

        FEFaceValues<dim, spacedim> fe_values(mapping,
                                              fe,
                                              exemplar_quadrature,
                                              update_flags);
        MechanicsValues<dim,
                        spacedim,
                        LinearAlgebra::distributed::Vector<double>>
          me_values(fe_values, current_position, current_velocity, me_flags);

        const unsigned int n_quadrature_points = exemplar_quadrature.size();
        std::vector<types::global_dof_index>     cell_dofs(fe.dofs_per_cell);
        std::vector<double>                      cell_rhs(fe.dofs_per_cell);
        std::vector<Tensor<1, spacedim, double>> one_force(n_quadrature_points);
        std::vector<Tensor<1, spacedim, double>> accumulated_forces(
          n_quadrature_points);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned() && cell->at_boundary())
            for (const auto &face_n : cell->face_indices())
              // only apply forces on physical boundaries
              if (!cell->has_periodic_neighbor(face_n) &&
                  cell->face(face_n)->at_boundary())
                {
                  cell->get_dof_indices(cell_dofs);
                  fe_values.reinit(cell, face_n);
                  me_values.reinit();
                  std::fill(accumulated_forces.begin(),
                            accumulated_forces.end(),
                            Tensor<1, spacedim, double>());
                  std::fill(cell_rhs.begin(), cell_rhs.end(), 0.0);
                  auto &extractor = fe_values[FEValuesExtractors::Vector(0)];

                  // Compute forces at quadrature points
                  for (const ForceContribution<dim, spacedim> *fc :
                       current_forces)
                    {
                      std::fill(one_force.begin(),
                                one_force.end(),
                                Tensor<1, spacedim, double>());
                      auto view =
                        make_array_view(one_force.begin(), one_force.end());
                      fc->compute_surface_force(time,
                                                me_values,
                                                cell->face(face_n),
                                                view);
                      for (unsigned int qp_n = 0; qp_n < n_quadrature_points;
                           ++qp_n)
                        accumulated_forces[qp_n] += one_force[qp_n];
                    }

                  // Assemble the RHS vector
                  //
                  // TODO - we could make this a lot faster by exploiting the
                  // fact that we have primitive FEs most of the time
                  for (unsigned int qp_n = 0; qp_n < n_quadrature_points;
                       ++qp_n)
                    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                      // F . phi dx
                      cell_rhs[i] += scalar_product(accumulated_forces[qp_n],
                                                    extractor.value(i, qp_n)) *
                                     fe_values.JxW(qp_n);

                  force_rhs.add(cell_dofs, cell_rhs);
                }
    } while (remaining_forces.size() > 0);
  }

  template void
  compute_volumetric_pk1_load_vector<NDIM - 1, NDIM>(
    const DoFHandler<NDIM - 1, NDIM> &,
    const Mapping<NDIM - 1, NDIM> &,
    const std::vector<ForceContribution<NDIM - 1, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);

  template void
  compute_volumetric_pk1_load_vector<NDIM, NDIM>(
    const DoFHandler<NDIM, NDIM> &,
    const Mapping<NDIM, NDIM> &,
    const std::vector<ForceContribution<NDIM, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);

  template void
  compute_volumetric_force_load_vector<NDIM - 1, NDIM>(
    const DoFHandler<NDIM - 1, NDIM> &,
    const Mapping<NDIM - 1, NDIM> &,
    const std::vector<ForceContribution<NDIM - 1, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);

  template void
  compute_boundary_force_load_vector<NDIM, NDIM>(
    const DoFHandler<NDIM, NDIM> &,
    const Mapping<NDIM, NDIM> &,
    const std::vector<ForceContribution<NDIM, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);

  template void
  compute_boundary_force_load_vector<NDIM - 1, NDIM>(
    const DoFHandler<NDIM - 1, NDIM> &,
    const Mapping<NDIM - 1, NDIM> &,
    const std::vector<ForceContribution<NDIM - 1, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);

  template void
  compute_volumetric_force_load_vector<NDIM, NDIM>(
    const DoFHandler<NDIM, NDIM> &,
    const Mapping<NDIM, NDIM> &,
    const std::vector<ForceContribution<NDIM, NDIM> *> &,
    const double,
    const LinearAlgebra::distributed::Vector<double> &,
    const LinearAlgebra::distributed::Vector<double> &,
    LinearAlgebra::distributed::Vector<double> &);
} // namespace fdl
