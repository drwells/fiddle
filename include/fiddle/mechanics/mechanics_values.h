#ifndef included_fiddle_mechanics_mechanics_values_h
#define included_fiddle_mechanics_mechanics_values_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Enum controlling update values. Computes all the useful things solid
   * mechanics codes might want to use. compute_load_vector() and related
   * functions use these flags to compute invariants in the least expensive
   * manner possible.
   *
   * In 2D, the computed values still correspond to 3D values: see, e.g.,
   * update_first_invariant.
   */
  enum MechanicsUpdateFlags
  {
    /**
     * Don't update any values.
     */
    update_nothing = 0x0000,

    /**
     * Update the deformation gradient (FF).
     */
    update_FF = 0x0001,

    /**
     * Update the transpose inverse of FF. Only valid when dim == spacedim.
     */
    update_FF_inv_T = 0x0002,

    /**
     * Update the determinant of FF. Only valid when dim == spacedim.
     */
    update_det_FF = 0x0004,

    /**
     * Update det(FF)^{-2/3}. Only valid when dim == spacedim.
     *
     * This value is intended to be used with the modified invariants.
     */
    update_n23_det_FF = 0x0008,

    /**
     * Update the positions at the quadrature points.
     */
    update_position_values = 0x0010,

    /**
     * Update the velocities at the quadrature points.
     */
    update_velocity_values = 0x0020,

    /**
     * The right Cauchy-Green deformation tensor: C := F^T F.
     */
    update_right_cauchy_green = 0x0040,

    /**
     * The first invariant: tr(C) in 3D. In 2D this is tr(C) + 1 to account for
     * the 'missing' row and column.
     */
    update_first_invariant = 0x0080,

    /**
     * The second invariant: 1/2(tr(C)^2 - tr(C^2)) in 3D. Like the first
     * invariant, this is 1/2(tr(C)^2 - tr(C^2)) + tr(C) in 2D to account for
     * the missing component.
     */
    update_second_invariant = 0x0100,

    /**
     * The third invariant: det(C). If dim == spacedim then this is also
     * det(FF)^2.
     */
    update_third_invariant = 0x0200,
  };

  // Manipulation routines for flags

  inline MechanicsUpdateFlags
  operator|(const MechanicsUpdateFlags f1, const MechanicsUpdateFlags f2)
  {
    return static_cast<MechanicsUpdateFlags>(static_cast<unsigned int>(f1) |
                                             static_cast<unsigned int>(f2));
  }

  inline MechanicsUpdateFlags &
  operator|=(MechanicsUpdateFlags &f1, const MechanicsUpdateFlags f2)
  {
    f1 = f1 | f2;
    return f1;
  }

  inline MechanicsUpdateFlags
  operator&(const MechanicsUpdateFlags f1, const MechanicsUpdateFlags f2)
  {
    return static_cast<MechanicsUpdateFlags>(static_cast<unsigned int>(f1) &
                                             static_cast<unsigned int>(f2));
  }

  inline MechanicsUpdateFlags &
  operator&=(MechanicsUpdateFlags &f1, const MechanicsUpdateFlags f2)
  {
    f1 = f1 & f2;
    return f1;
  }

  inline UpdateFlags
  compute_flag_dependencies(const MechanicsUpdateFlags me_flags)
  {
    UpdateFlags flags = UpdateFlags::update_default;

    if (me_flags & update_FF)
      flags |= update_gradients;
    if (me_flags & update_FF_inv_T)
      flags |= update_gradients;
    if (me_flags & update_det_FF)
      flags |= update_gradients;
    if (me_flags & update_position_values)
      flags |= update_values;
    if (me_flags & update_velocity_values)
      flags |= update_values;

    return flags;
  }

  inline MechanicsUpdateFlags
  resolve_flag_dependencies(const MechanicsUpdateFlags me_flags)
  {
    MechanicsUpdateFlags result = me_flags;
    // Resolve dependencies of dependencies by iteration. We only need three
    // iterations - do 4 out of an abundance of caution
    for (unsigned int i = 0; i < 4; ++i)
      {
        if (result & update_FF_inv_T)
          result |= update_FF;
        if (result & update_det_FF)
          result |= update_FF;
        if (result & update_n23_det_FF)
          result |= update_det_FF;
        if (result & update_right_cauchy_green)
          result |= update_FF;
        if (result & update_first_invariant)
          // needs tr(C)
          result |= update_right_cauchy_green;
        if (result & update_second_invariant)
          // needs tr(C) and tr(C^2)
          result |= update_first_invariant;
        if (result & update_third_invariant)
          // needs det(C)
          // TODO: make this work when dim != spacedim
          result |= update_det_FF;
      }

    return result;
  }

  /**
   * Class that computes mechanics values and other things we need for
   * evaluating stress functions.
   */
  template <int dim,
            int spacedim        = dim,
            typename VectorType = LinearAlgebra::distributed::Vector<double>>
  class MechanicsValues
  {
  public:
    MechanicsValues(const FEValuesBase<dim, spacedim> &fe_values,
                    const VectorType &                 position,
                    const VectorType &                 velocity,
                    const MechanicsUpdateFlags         flags);

    template <typename Iterator>
    void
    reinit(const Iterator &cell);

    const FEValuesBase<dim, spacedim> &
    get_fe_values() const;

    const std::vector<Tensor<2, spacedim>> &
    get_FF() const;

    const std::vector<Tensor<2, spacedim>> &
    get_FF_inv_T() const;

    const std::vector<double> &
    get_det_FF() const;

    const std::vector<double> &
    get_n23_det_FF() const;

    const std::vector<Tensor<1, spacedim>> &
    get_position_values() const;

    const std::vector<Tensor<1, spacedim>> &
    get_velocity_values() const;

    const std::vector<SymmetricTensor<2, spacedim>> &
    get_right_cauchy_green() const;

    const std::vector<double> &
    get_first_invariant() const;

    const std::vector<double> &
    get_second_invariant() const;

    const std::vector<double> &
    get_third_invariant() const;

  protected:
    SmartPointer<const FEValuesBase<dim, spacedim>> fe_values;

    SmartPointer<const VectorType> position;

    SmartPointer<const VectorType> velocity;

    MechanicsUpdateFlags update_flags;

    std::vector<Tensor<2, spacedim>> FF;

    std::vector<Tensor<2, spacedim>> FF_inv_T;

    std::vector<double> det_FF;

    std::vector<double> n23_det_FF;

    std::vector<Tensor<1, spacedim>> position_values;

    std::vector<Tensor<1, spacedim>> velocity_values;

    std::vector<SymmetricTensor<2, spacedim>> right_cauchy_green;

    std::vector<double> first_invariant;

    std::vector<double> second_invariant;

    std::vector<double> third_invariant;

    std::vector<types::global_dof_index> scratch_dof_indices;

    std::vector<double> scratch_position_values;

    std::vector<double> scratch_velocity_values;
  };

  // Constructor and reinitialization

  template <int dim, int spacedim, typename VectorType>
  MechanicsValues<dim, spacedim, VectorType>::MechanicsValues(
    const FEValuesBase<dim, spacedim> &fe_values,
    const VectorType &                 position,
    const VectorType &                 velocity,
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

    if (update_flags & MechanicsUpdateFlags::update_FF)
      FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_FF_inv_T)
      FF_inv_T.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_det_FF)
      det_FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_n23_det_FF)
      n23_det_FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_position_values)
      position_values.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_velocity_values)
      velocity_values.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_right_cauchy_green)
      right_cauchy_green.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_first_invariant)
      first_invariant.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_second_invariant)
      second_invariant.resize(this->fe_values->n_quadrature_points);
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

  template <int dim, int spacedim, typename VectorType>
  inline const FEValuesBase<dim, spacedim> &
  MechanicsValues<dim, spacedim, VectorType>::get_fe_values() const
  {
    return *fe_values;
  }

  // Access functions

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_FF() const
  {
    Assert(update_flags & update_FF, ExcMessage("Needs update_FF"));
    return FF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_FF_inv_T() const
  {
    Assert(update_flags & update_FF_inv_T, ExcMessage("Needs update_FF_inv_T"));
    return FF_inv_T;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_det_FF() const
  {
    Assert(update_flags & update_det_FF, ExcMessage("Needs update_det_FF"));
    return det_FF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_n23_det_FF() const
  {
    Assert(update_flags & update_n23_det_FF,
           ExcMessage("Needs update_n23_det_FF"));
    return n23_det_FF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<1, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_position_values() const
  {
    Assert(update_flags & update_position_values,
           ExcMessage("Needs update_position_values"));
    return position_values;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<1, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_velocity_values() const
  {
    Assert(update_flags & update_velocity_values,
           ExcMessage("Needs update_velocity_values"));
    return velocity_values;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<SymmetricTensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_right_cauchy_green() const
  {
    Assert(update_flags & update_right_cauchy_green,
           ExcMessage("Needs update_right_cauchy_green"));
    return right_cauchy_green;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_first_invariant() const
  {
    Assert(update_flags & update_first_invariant,
           ExcMessage("Needs update_first_invariant"));
    return first_invariant;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_second_invariant() const
  {
    Assert(update_flags & update_second_invariant,
           ExcMessage("Needs update_second_invariant"));
    return second_invariant;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_third_invariant() const
  {
    Assert(update_flags & update_third_invariant,
           ExcMessage("Needs update_third_invariant"));
    return third_invariant;
  }
} // namespace fdl

#endif
