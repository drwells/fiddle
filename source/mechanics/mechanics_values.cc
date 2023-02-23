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

  namespace
  {
    /**
     * Resolve the interdependencies within a single MechanicsUpdateFlags
     * object.
     */
    MechanicsUpdateFlags
    resolve_flag_dependencies(const MechanicsUpdateFlags me_flags)
    {
      MechanicsUpdateFlags result = me_flags;
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
          if (result & update_deformed_normal_vectors)
            result |= update_FF_inv_T;
          if (result & update_right_cauchy_green)
            result |= update_FF;
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
        }
      while (old_result != result);

      return result;
    }
  } // namespace

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


  // Constructor and reinitialization

  template <int dim, int spacedim, typename VectorType>
  MechanicsValues<dim, spacedim, VectorType>::MechanicsValues(
    const FEValuesBase<dim, spacedim> &fe_values,
    const VectorType                  &position,
    const VectorType                  &velocity,
    const MechanicsUpdateFlags         flags)
    : fe_values(&fe_values)
    , position(&position)
    , velocity(&velocity)
    , update_flags(flags)
  {
    // Resolve flag dependencies:
    update_flags = resolve_flag_dependencies(update_flags);

    // Check some things:
    if (update_flags & update_FF)
      {
        Assert(this->fe_values->get_update_flags() &
                 UpdateFlags::update_gradients,
               ExcMessage("This class needs gradients"));
      }
    if ((update_flags & update_position_values) ||
        (update_flags & update_velocity_values))
      {
        Assert(this->fe_values->get_update_flags() & UpdateFlags::update_values,
               ExcMessage("This class needs values"));
      }

    if ((update_flags & update_deformed_normal_vectors))
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

    // basic terms dependent on the deformation gradient:
    if (update_flags & MechanicsUpdateFlags::update_FF)
      FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_FF_inv_T)
      FF_inv_T.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_det_FF)
      det_FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_n23_det_FF)
      n23_det_FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_right_cauchy_green)
      right_cauchy_green.resize(this->fe_values->n_quadrature_points);

    // physical values:
    if (update_flags & MechanicsUpdateFlags::update_position_values)
      position_values.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_deformed_normal_vectors)
      deformed_normal_vectors.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_velocity_values)
      velocity_values.resize(this->fe_values->n_quadrature_points);

    // invariants:
    if (update_flags & MechanicsUpdateFlags::update_first_invariant)
      first_invariant.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_modified_first_invariant)
      modified_first_invariant.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_second_invariant)
      second_invariant.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_modified_second_invariant)
      modified_second_invariant.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_third_invariant)
      third_invariant.resize(this->fe_values->n_quadrature_points);
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

    for (unsigned int q = 0; q < fe_values->n_quadrature_points; ++q)
      {
        SymmetricTensor<2, spacedim> temp;
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
        if (update_flags & update_deformed_normal_vectors)
          {
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
          modified_second_invariant[q] = second_invariant[q]
            * std::pow(n23_det_FF[q], 2);
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
      }

    if (update_flags & update_position_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_position_values, position_values);

    if (update_flags & update_velocity_values)
      (*fe_values)[vec].get_function_values_from_local_dof_values(
        scratch_velocity_values, velocity_values);
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
