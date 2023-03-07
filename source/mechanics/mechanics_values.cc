#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace fdl
{
  using namespace dealii;

  MechanicsUpdateFlags
  resolve_flag_dependencies(const MechanicsUpdateFlags me_flags)
  {
    MechanicsUpdateFlags result     = me_flags;
    MechanicsUpdateFlags old_result = me_flags;
    // iterate until convergence
    do
      {
        old_result = result;
        if (result & update_FF_inv_T)
          result |= update_FF;
        if (result & update_det_FF)
          result |= update_FF;
        if (result & update_n23_det_FF)
          result |= update_det_FF;
        if (result & update_log_det_FF)
          result |= update_det_FF;
        if (result & update_deformed_normal_vectors)
          result |= update_FF_inv_T;
        if (result & update_right_cauchy_green)
          result |= update_FF;
        if (result & update_green)
          result |= update_right_cauchy_green;
        if (result & update_first_invariant)
          // needs tr(C)
          result |= update_right_cauchy_green;
        if (result & update_modified_first_invariant)
          result |= update_first_invariant | update_n23_det_FF;
        if (result & update_modified_second_invariant)
          result |= update_second_invariant | update_n23_det_FF;
        if (result & update_second_invariant)
          // needs tr(C) and tr(C^2)
          result |= update_first_invariant;
        if (result & update_third_invariant)
          // needs det(C)
          // TODO: make this work when dim != spacedim
          result |= update_det_FF;
        if (result & update_first_invariant_dFF)
          result |= update_FF;
        if (result & update_modified_first_invariant_dFF)
          result |= update_n23_det_FF | update_FF | update_first_invariant |
                    update_FF_inv_T;
    } while (old_result != result);

    return result;
  }

  UpdateFlags
  compute_flag_dependencies(const MechanicsUpdateFlags me_flags)
  {
    MechanicsUpdateFlags actual_flags = resolve_flag_dependencies(me_flags);
    UpdateFlags          flags        = UpdateFlags::update_default;

    if (actual_flags & update_FF)
      flags |= update_gradients;
    if (actual_flags & update_FF_inv_T)
      flags |= update_gradients;
    if (actual_flags & update_det_FF)
      flags |= update_gradients;
    if (actual_flags & update_deformed_normal_vectors)
      flags |= update_normal_vectors;
    if (actual_flags & update_position_values)
      flags |= update_values;
    if (actual_flags & update_velocity_values)
      flags |= update_values;

    return flags;
  }

  // Constructors and reinitialization

  template <int dim, int spacedim, typename VectorType>
  MechanicsValues<dim, spacedim, VectorType>::MechanicsValues(
    const FEValuesBase<dim, spacedim> &fe_values,
    const VectorType                  &position,
    const VectorType                  &velocity,
    const MechanicsUpdateFlags         flags)
    : fe_values(&fe_values)
    , position(&position)
    , velocity(&velocity)
    , update_flags(resolve_flag_dependencies(flags))
  {
    // Check that FEValues has all the flags we need:
    const auto required_flags =
      static_cast<unsigned int>(compute_flag_dependencies(update_flags));
    const auto provided_flags =
      static_cast<unsigned int>(fe_values.get_update_flags());
    for (unsigned int b = 0; b < 8 * sizeof(UpdateFlags); ++b)
      {
        const bool required_set = ((required_flags >> b) & 1U) != 0u;
        const bool provided_set = ((provided_flags >> b) & 1U) != 0u;

        if (required_set && !provided_set)
          {
            std::ostringstream required_str;
            required_str << compute_flag_dependencies(update_flags);
            std::ostringstream provided_str;
            required_str << fe_values.get_update_flags();
            AssertThrow(
              false,
              ExcMessage("The provided update flags '" + provided_str.str() +
                         "' do not contain all of the required update flags '" +
                         required_str.str() + "'."));
          }
      }

    if (update_flags & update_deformed_normal_vectors)
      {
        AssertThrow(
          (dynamic_cast<const FEFaceValues<dim, spacedim> *>(&fe_values) !=
           nullptr),
          ExcMessage(
            "Normal vectors can only be requested with face integration."));
      }

    // Set up arrays:
    const auto n_dofs_per_cell = this->fe_values->get_fe().n_dofs_per_cell();
    if (update_flags & MechanicsUpdateFlags::update_position_values ||
        update_flags & MechanicsUpdateFlags::update_FF)
      {
        scratch_position_values.resize(n_dofs_per_cell);
        scratch_dof_indices.resize(n_dofs_per_cell);
      }

    if (update_flags & MechanicsUpdateFlags::update_velocity_values)
      {
        scratch_velocity_values.resize(n_dofs_per_cell);
        scratch_dof_indices.resize(n_dofs_per_cell);
      }

    resize(this->fe_values->n_quadrature_points);
  }

  template <int dim, int spacedim, typename VectorType>
  MechanicsValues<dim, spacedim, VectorType>::MechanicsValues(
    const MechanicsUpdateFlags flags)
    : update_flags(resolve_flag_dependencies(flags))
  {
    AssertThrow(!(update_flags & update_velocity_values),
                ExcMessage("This ctor cannot be used to compute velocities."));
    AssertThrow(!(update_flags & update_position_values),
                ExcMessage("This ctor cannot be used to compute positions."));
    AssertThrow(!(update_flags & update_deformed_normal_vectors),
                ExcMessage(
                  "This ctor cannot be used to compute deformed normal "
                  "vectors."));
  }


  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::resize(const std::size_t size)
  {
    // basic terms dependent on the deformation gradient:
    if (update_flags & MechanicsUpdateFlags::update_FF)
      {
        FF.resize(size);
        scratch_FF.resize(size);
      }
    if (update_flags & MechanicsUpdateFlags::update_FF_inv_T)
      FF_inv_T.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_det_FF)
      det_FF.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_n23_det_FF)
      n23_det_FF.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_log_det_FF)
      log_det_FF.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_right_cauchy_green)
      right_cauchy_green.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_green)
      green.resize(size);

    // physical values:
    if (update_flags & MechanicsUpdateFlags::update_position_values)
      position_values.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_deformed_normal_vectors)
      deformed_normal_vectors.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_velocity_values)
      velocity_values.resize(size);

    // invariants:
    if (update_flags & MechanicsUpdateFlags::update_first_invariant)
      first_invariant.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_modified_first_invariant)
      modified_first_invariant.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_second_invariant)
      second_invariant.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_modified_second_invariant)
      modified_second_invariant.resize(size);
    if (update_flags & MechanicsUpdateFlags::update_third_invariant)
      third_invariant.resize(size);

    // derivatives of invariants:
    if (update_flags & MechanicsUpdateFlags::update_first_invariant_dFF)
      first_invariant_dFF.resize(size);
    if (update_flags &
        MechanicsUpdateFlags::update_modified_first_invariant_dFF)
      modified_first_invariant_dFF.resize(size);
  }

  template <int dim, int spacedim, typename VectorType>
  template <typename Iterator>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit(const Iterator &cell)
  {
    static_assert(
      std::is_same<
        Iterator,
        typename DoFHandler<dim, spacedim>::active_cell_iterator>::value ||
        std::is_same<
          Iterator,
          typename DoFHandler<dim, spacedim>::active_face_iterator>::value,
      "The only supported iterator types are active cell and face DoFHandler "
      "iterators.");
    Assert(fe_values,
           ExcMessage(
             "This function can only be called when the present object is set "
             "up to use an FEValues object."));
    Assert(cell->level() == fe_values->get_cell()->level() &&
             cell->index() == fe_values->get_cell()->index(),
           ExcMessage("The provided cell must be the same as the one used in "
                      "the corresponding FEValues object."));

    const FEValuesExtractors::Vector vec(0);

    const bool update_scratch_positions =
      update_flags & update_FF || update_flags & update_position_values;
    const bool update_scratch_velocities =
      update_flags & update_velocity_values;

    if (update_scratch_positions || update_scratch_velocities)
      cell->get_dof_indices(scratch_dof_indices);

    if (update_scratch_positions)
      for (unsigned int i = 0; i < scratch_dof_indices.size(); ++i)
        scratch_position_values[i] = (*position)[scratch_dof_indices[i]];

    if (update_scratch_velocities)
      for (unsigned int i = 0; i < scratch_dof_indices.size(); ++i)
        scratch_velocity_values[i] = (*velocity)[scratch_dof_indices[i]];

    if (update_flags & update_FF)
      (*fe_values)[vec].get_function_gradients_from_local_dof_values(
        scratch_position_values, FF);

    reinit_from_FF();

    if (update_flags & update_position_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_position_values, position_values);

    if (update_flags & update_velocity_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_velocity_values, velocity_values);
  }

  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const ActiveStrain<dim, spacedim> &active_strain)
  {
    Assert(fe_values,
           ExcMessage(
             "This function can only be called when the present object is set "
             "up to use an FEValues object."));
    Assert(cell->level() == fe_values->get_cell()->level() &&
             cell->index() == fe_values->get_cell()->index(),
           ExcMessage("The provided cell must be the same as the one used in "
                      "the corresponding FEValues object."));

    const FEValuesExtractors::Vector vec(0);

    const bool update_scratch_positions =
      update_flags & update_FF || update_flags & update_position_values;
    const bool update_scratch_velocities =
      update_flags & update_velocity_values;

    if (update_scratch_positions || update_scratch_velocities)
      cell->get_dof_indices(scratch_dof_indices);

    if (update_scratch_positions)
      for (unsigned int i = 0; i < scratch_dof_indices.size(); ++i)
        scratch_position_values[i] = (*position)[scratch_dof_indices[i]];

    if (update_scratch_velocities)
      for (unsigned int i = 0; i < scratch_dof_indices.size(); ++i)
        scratch_velocity_values[i] = (*velocity)[scratch_dof_indices[i]];

    if (update_flags & update_FF)
      {
        (*fe_values)[vec].get_function_gradients_from_local_dof_values(
          scratch_position_values, scratch_FF);

        auto view = make_array_view(FF);
        active_strain.push_deformation_gradient_forward(
          cell, make_array_view(scratch_FF), view);
      }

    reinit_from_FF();

    if (update_flags & update_position_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_position_values, position_values);

    if (update_flags & update_velocity_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_velocity_values, velocity_values);
  }


  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit(
    const std::vector<Tensor<2, spacedim>> &provided_FF)
  {
    Assert(!(update_flags & update_velocity_values),
           ExcMessage("This reinit() function cannot be used to compute "
                      "velocities."));
    Assert(!(update_flags & update_position_values),
           ExcMessage("This reinit() function cannot be used to compute "
                      "positions."));
    Assert(!(update_flags & update_deformed_normal_vectors),
           ExcMessage("This reinit() function cannot be used to compute "
                      "deformed normal vectors."));

    // Probably not needed, but who knows where this preprocessor is getting its
    // data from
    if (provided_FF.size() != FF.size())
      resize(provided_FF.size());

    FF = provided_FF;

    reinit_from_FF();
  }

  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit_from_FF()
  {
    for (unsigned int q = 0; q < FF.size(); ++q)
      {
        if (update_flags & update_FF_inv_T)
          FF_inv_T[q] = transpose(invert(FF[q]));
        if (update_flags & update_det_FF)
          det_FF[q] = determinant(FF[q]);
        if (update_flags & update_n23_det_FF)
          {
            // It is slightly more accurate (according to herbie) to do
            // division, cbrt, and then multiply
            const auto temp = std::cbrt(1.0 / det_FF[q]);
            n23_det_FF[q]   = temp * temp;
          }
        if (update_flags & update_log_det_FF)
          {
            log_det_FF[q] = std::log(det_FF[q]);
          }
        if (update_flags & update_deformed_normal_vectors)
          {
            Assert(fe_values, ExcFDLInternalError());
            deformed_normal_vectors[q] =
              FF_inv_T[q] * fe_values->normal_vector(q);
            deformed_normal_vectors[q] /= deformed_normal_vectors[q].norm();
          }
        if (update_flags & update_right_cauchy_green)
          // TODO - get rid of the call to symmetrize()
          {
            Assert(update_flags & update_FF, ExcFDLInternalError());
            right_cauchy_green[q] = SymmetricTensor<2, spacedim>(
              symmetrize(transpose(FF[q]) * FF[q]));
          }
        if (update_flags & update_green)
          {
            Assert(update_flags & update_green, ExcFDLInternalError());
            green[q] = right_cauchy_green[q];
            for (unsigned int d = 0; d < spacedim; ++d)
              green[q][d][d] -= 1.0;
            green[q] *= 0.5;
          }

        // invariants
        if (update_flags & update_first_invariant)
          {
            Assert(update_flags & update_right_cauchy_green,
                   ExcFDLInternalError());
            Assert(dim == 2 || dim == 3, ExcFDLInternalError());
            if (dim == 2)
              first_invariant[q] =
                dealii::first_invariant(right_cauchy_green[q]) + 1.0;
            else
              first_invariant[q] =
                dealii::first_invariant(right_cauchy_green[q]);
          }
        if (update_flags & update_modified_first_invariant)
          modified_first_invariant[q] = first_invariant[q] * n23_det_FF[q];
        if (update_flags & update_second_invariant)
          {
            Assert(update_flags & update_right_cauchy_green,
                   ExcFDLInternalError());
            Assert(dim == 2 || dim == 3, ExcFDLInternalError());
            if (dim == 2)
              second_invariant[q] =
                dealii::second_invariant(right_cauchy_green[q]) +
                trace(right_cauchy_green[q]);
            else
              second_invariant[q] =
                dealii::second_invariant(right_cauchy_green[q]);
          }
        if (update_flags & update_modified_second_invariant)
          modified_second_invariant[q] =
            second_invariant[q] * std::pow(n23_det_FF[q], 2);
        if (update_flags & update_third_invariant)
          {
            if (dim == spacedim)
              {
                Assert(update_flags & update_det_FF, ExcFDLInternalError());
                third_invariant[q] = det_FF[q] * det_FF[q];
              }
            else
              {
                Assert(update_flags & update_right_cauchy_green,
                       ExcFDLInternalError());
                third_invariant[q] =
                  dealii::third_invariant(right_cauchy_green[q]);
              }
          }

        // derivatives of invariants
        if (update_flags & update_first_invariant_dFF)
          first_invariant_dFF[q] = 2.0 * FF[q];

        if (update_flags & update_modified_first_invariant_dFF)
          modified_first_invariant_dFF[q] =
            2.0 * n23_det_FF[q] *
            (FF[q] - (1.0 / 3.0) * first_invariant[q] * FF_inv_T[q]);
      }
  }


  template class MechanicsValues<NDIM - 1, NDIM, Vector<double>>;
  template class MechanicsValues<NDIM, NDIM, Vector<double>>;

  template class MechanicsValues<NDIM - 1,
                                 NDIM,
                                 LinearAlgebra::distributed::Vector<double>>;
  template class MechanicsValues<NDIM,
                                 NDIM,
                                 LinearAlgebra::distributed::Vector<double>>;

  template void
  MechanicsValues<NDIM - 1, NDIM, Vector<double>>::reinit(
    const DoFHandler<NDIM - 1, NDIM>::active_cell_iterator &cell);
  template void
  MechanicsValues<NDIM - 1, NDIM, Vector<double>>::reinit(
    const DoFHandler<NDIM - 1, NDIM>::active_face_iterator &cell);

  template void
  MechanicsValues<NDIM, NDIM, Vector<double>>::reinit(
    const DoFHandler<NDIM, NDIM>::active_cell_iterator &cell);
  template void
  MechanicsValues<NDIM, NDIM, Vector<double>>::reinit(
    const DoFHandler<NDIM, NDIM>::active_face_iterator &cell);

  template void
  MechanicsValues<NDIM - 1, NDIM, LinearAlgebra::distributed::Vector<double>>::
    reinit(const DoFHandler<NDIM - 1, NDIM>::active_cell_iterator &cell);
  template void
  MechanicsValues<NDIM - 1, NDIM, LinearAlgebra::distributed::Vector<double>>::
    reinit(const DoFHandler<NDIM - 1, NDIM>::active_face_iterator &cell);

  template void
  MechanicsValues<NDIM, NDIM, LinearAlgebra::distributed::Vector<double>>::
    reinit(const DoFHandler<NDIM, NDIM>::active_cell_iterator &cell);
  template void
  MechanicsValues<NDIM, NDIM, LinearAlgebra::distributed::Vector<double>>::
    reinit(const DoFHandler<NDIM, NDIM>::active_face_iterator &cell);
} // namespace fdl
