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

    std::vector<types::material_id>
    setup_material_ids(const std::vector<types::material_id> &material_ids)
    {
      std::vector<types::material_id> result = material_ids;
      // If the user doesn't want any material ids, let them do it. This utility
      // function is only for the explicit material id case.
      if (result.size() == 0)
        result.push_back(numbers::invalid_material_id);

      // permit duplicates in the input array
      std::sort(result.begin(), result.end());
      result.erase(std::unique(result.begin(), result.end()), result.end());

      return result;
    }
  } // namespace

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                            &quad,
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
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>           &quad,
    const double                     spring_constant,
    const DoFHandler<dim, spacedim> &dof_handler,
    const Mapping<dim, spacedim>    &mapping,
    const Function<spacedim>        &reference_position)
    : ForceContribution<dim, spacedim, double>(quad)
    , spring_constant(spring_constant)
    , dof_handler(&dof_handler)
    , reference_position(
        do_interpolation(dof_handler, mapping, reference_position))
  {
    this->reference_position.update_ghost_values();
  }

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                            &quad,
    const double                                      spring_constant,
    const DoFHandler<dim, spacedim>                  &dof_handler,
    const std::vector<types::material_id>            &material_ids,
    const LinearAlgebra::distributed::Vector<double> &reference_position)
    : ForceContribution<dim, spacedim, double>(quad)
    , material_ids(setup_material_ids(material_ids))
    , spring_constant(spring_constant)
    , dof_handler(&dof_handler)
    , reference_position(reference_position)
  {
    this->reference_position.update_ghost_values();
  }

  template <int dim, int spacedim, typename Number>
  SpringForce<dim, spacedim, Number>::SpringForce(
    const Quadrature<dim>                 &quad,
    const double                           spring_constant,
    const DoFHandler<dim, spacedim>       &dof_handler,
    const Mapping<dim, spacedim>          &mapping,
    const std::vector<types::material_id> &material_ids,
    const Function<spacedim>              &reference_position)
    : ForceContribution<dim, spacedim, double>(quad)
    , material_ids(setup_material_ids(material_ids))
    , spring_constant(spring_constant)
    , dof_handler(&dof_handler)
    , reference_position(
        do_interpolation(dof_handler, mapping, reference_position))
  {
    this->reference_position.update_ghost_values();
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForce<dim, spacedim, Number>::set_reference_position(
    const LinearAlgebra::distributed::Vector<double> &reference_position)
  {
    this->reference_position = reference_position;
    this->reference_position.update_ghost_values();
  }


  template <int dim, int spacedim, typename Number>
  MechanicsUpdateFlags
  SpringForce<dim, spacedim, Number>::get_mechanics_update_flags() const
  {
    return MechanicsUpdateFlags::update_nothing;
  }

  template <int dim, int spacedim, typename Number>
  UpdateFlags
  SpringForce<dim, spacedim, Number>::get_update_flags() const
  {
    return UpdateFlags::update_values;
  }

  template <int dim, int spacedim, typename Number>
  bool
  SpringForce<dim, spacedim, Number>::is_volume_force() const
  {
    return true;
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForce<dim, spacedim, Number>::setup_force(
    const double /*time*/,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> & /*velocity*/)
  {
    current_position = &position;
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForce<dim, spacedim, Number>::finish_force(const double /*time*/)
  {
    current_position = nullptr;
  }

  template <int dim, int spacedim, typename Number>
  void
  SpringForce<dim, spacedim, Number>::compute_volume_force(
    const double /*time*/,
    const MechanicsValues<dim, spacedim>                              &m_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    ArrayView<Tensor<1, spacedim, Number>> &forces) const
  {
    if (material_ids.size() > 0 && !std::binary_search(material_ids.begin(),
                                                       material_ids.end(),
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
            &dof_handler->get_triangulation(),
            cell->level(),
            cell->index(),
            &*dof_handler);

        scratch_cell_dofs.resize(fe_values.dofs_per_cell);
        dof_cell->get_dof_indices(scratch_cell_dofs);
        scratch_dof_values.resize(fe_values.dofs_per_cell);
        scratch_qp_values.resize(fe_values.n_quadrature_points);

        auto &extractor = fe_values[FEValuesExtractors::Vector(0)];
        for (unsigned int i = 0; i < scratch_cell_dofs.size(); ++i)
          scratch_dof_values[i] =
            spring_constant * (reference_position[scratch_cell_dofs[i]] -
                               (*current_position)[scratch_cell_dofs[i]]);
        extractor.get_function_values_from_local_dof_values(scratch_dof_values,
                                                            scratch_qp_values);
        std::copy(scratch_qp_values.begin(),
                  scratch_qp_values.end(),
                  forces.begin());
      }
  }



  template class SpringForce<NDIM - 1, NDIM, double>;
  template class SpringForce<NDIM, NDIM, double>;
} // namespace fdl
