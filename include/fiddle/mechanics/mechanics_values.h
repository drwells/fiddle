#ifndef included_fiddle_mechanics_mechanics_values_h
#define included_fiddle_mechanics_mechanics_values_h

#include <fiddle/base/config.h>

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>

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
     * Update log(det(FF)). Only valid when dim == spacedim. Here log() is the
     * natural logarithm.
     */
    update_log_det_FF = 0x0010,

    /**
     * Update the normal vectors in the deformed configuration. Only available
     * for surface elements.
     */
    update_deformed_normal_vectors = 0x0020,

    /**
     * Update the positions at the quadrature points.
     */
    update_position_values = 0x0040,

    /**
     * Update the velocities at the quadrature points.
     */
    update_velocity_values = 0x0080,

    /**
     * The right Cauchy-Green deformation tensor: C := F^T F.
     */
    update_right_cauchy_green = 0x0100,

    /**
     * The symmetric Green strain tensor: E := 1/2 (C - I).
     */
    update_green = 0x0200,

    /**
     * The first invariant: tr(C) in 3D. In 2D this is tr(C) + 1 to account for
     * the 'missing' row and column.
     */
    update_first_invariant = 0x0400,

    /**
     * The modified first invariant: J^(-2/3) I1.
     */
    update_modified_first_invariant = 0x0800,

    /**
     * The second invariant: 1/2(tr(C)^2 - tr(C^2)) in 3D. Like the first
     * invariant, this is 1/2(tr(C)^2 - tr(C^2)) + tr(C) in 2D to account for
     * the missing component.
     */
    update_second_invariant = 0x1000,

    /**
     * The modified second invariant: J^(-4/3) I2.
     */
    update_modified_second_invariant = 0x2000,

    /**
     * The third invariant: det(C). If dim == spacedim then this is also
     * det(FF)^2.
     */
    update_third_invariant = 0x4000,

    /**
     * The derivative of the first invariant with respect to FF.
     */
    update_first_invariant_dFF = 0x8000,

    /**
     * The derivative of the modified first invariant with respect to FF.
     */
    update_modified_first_invariant_dFF = 0x0001'0000,
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

  /**
   * Compute the dependencies of a set of mechanics update flags necessary to
   * pass to an FEValues object.
   */
  UpdateFlags
  compute_flag_dependencies(const MechanicsUpdateFlags me_flags);

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
    /**
     * Constructor. @p fe_values must be set up with an UpdateFlags compatible
     * with @p MechanicsUpdateFlags - this is typically computed via
     * compute_flag_dependencies().
     */
    MechanicsValues(const FEValuesBase<dim, spacedim> &fe_values,
                    const VectorType                  &position,
                    const VectorType                  &velocity,
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

    const std::vector<double> &
    get_log_det_FF() const;

    const std::vector<Tensor<1, spacedim>> &
    get_deformed_normal_vectors() const;

    const std::vector<Tensor<1, spacedim>> &
    get_position_values() const;

    const std::vector<Tensor<1, spacedim>> &
    get_velocity_values() const;

    const std::vector<SymmetricTensor<2, spacedim>> &
    get_right_cauchy_green() const;

    const std::vector<SymmetricTensor<2, spacedim>> &
    get_green() const;

    const std::vector<double> &
    get_first_invariant() const;

    const std::vector<double> &
    get_modified_first_invariant() const;

    const std::vector<double> &
    get_second_invariant() const;

    const std::vector<double> &
    get_modified_second_invariant() const;

    const std::vector<double> &
    get_third_invariant() const;

    const std::vector<Tensor<2, spacedim>> &
    get_first_invariant_dFF() const;

    const std::vector<Tensor<2, spacedim>> &
    get_modified_first_invariant_dFF() const;

  protected:
    SmartPointer<const FEValuesBase<dim, spacedim>> fe_values;

    SmartPointer<const VectorType> position;

    SmartPointer<const VectorType> velocity;

    MechanicsUpdateFlags update_flags;

    std::vector<Tensor<2, spacedim>> FF;

    std::vector<Tensor<2, spacedim>> FF_inv_T;

    std::vector<double> det_FF;

    std::vector<double> n23_det_FF;

    std::vector<double> log_det_FF;

    std::vector<Tensor<1, spacedim>> deformed_normal_vectors;

    std::vector<Tensor<1, spacedim>> position_values;

    std::vector<Tensor<1, spacedim>> velocity_values;

    std::vector<SymmetricTensor<2, spacedim>> right_cauchy_green;

    std::vector<SymmetricTensor<2, spacedim>> green;

    std::vector<double> first_invariant;

    std::vector<double> modified_first_invariant;

    std::vector<double> second_invariant;

    std::vector<double> modified_second_invariant;

    std::vector<double> third_invariant;

    std::vector<Tensor<2, spacedim>> first_invariant_dFF;

    std::vector<Tensor<2, spacedim>> modified_first_invariant_dFF;

    std::vector<types::global_dof_index> scratch_dof_indices;

    std::vector<double> scratch_position_values;

    std::vector<double> scratch_velocity_values;
  };

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
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_log_det_FF() const
  {
    Assert(update_flags & update_log_det_FF,
           ExcMessage("Needs update_log_det_FF"));
    return log_det_FF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<1, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_deformed_normal_vectors()
    const
  {
    Assert(update_flags & update_deformed_normal_vectors,
           ExcMessage("Needs update_deformed_normal_vectors"));
    return deformed_normal_vectors;
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
  inline const std::vector<SymmetricTensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_green() const
  {
    Assert(update_flags & update_green, ExcMessage("Needs update_green"));
    return green;
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
  MechanicsValues<dim, spacedim, VectorType>::get_modified_first_invariant()
    const
  {
    Assert(update_flags & update_modified_first_invariant,
           ExcMessage("Needs update_modified_first_invariant"));
    return modified_first_invariant;
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
  MechanicsValues<dim, spacedim, VectorType>::get_modified_second_invariant()
    const
  {
    Assert(update_flags & update_modified_second_invariant,
           ExcMessage("Needs update_modified_second_invariant"));
    return modified_second_invariant;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_third_invariant() const
  {
    Assert(update_flags & update_third_invariant,
           ExcMessage("Needs update_third_invariant"));
    return third_invariant;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_first_invariant_dFF() const
  {
    Assert(update_flags & update_first_invariant_dFF,
           ExcMessage("Needs update_first_invariant_dFF"));
    return first_invariant_dFF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_modified_first_invariant_dFF()
    const
  {
    Assert(update_flags & update_modified_first_invariant_dFF,
           ExcMessage("Needs update_modified_first_invariant_dFF"));
    return modified_first_invariant_dFF;
  }
} // namespace fdl

#endif
