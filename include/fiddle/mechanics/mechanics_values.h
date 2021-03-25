#ifndef included_fiddle_mechanics_mechanics_values_h
#define included_fiddle_mechanics_mechanics_values_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  // Enum controlling update values

  enum MechanicsUpdateFlags
  {
    update_nothing         = 0x0000,
    update_FF              = 0x0001,
    update_FF_inv_T        = 0x0002,
    update_det_FF          = 0x0004,
    update_position_values = 0x0008,
    update_velocity_values = 0x0010
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

    void
    reinit();

    const FEValuesBase<dim, spacedim> &
    get_fe_values() const;

    const std::vector<Tensor<2, spacedim>> &
    get_FF() const;

    const std::vector<Tensor<2, spacedim>> &
    get_FF_inv_T() const;

    const std::vector<double> &
    get_det_FF() const;

    const std::vector<Tensor<1, spacedim>> &
    get_position_values() const;

    const std::vector<Tensor<1, spacedim>> &
    get_velocity_values() const;

  protected:
    SmartPointer<const FEValuesBase<dim, spacedim>> fe_values;

    SmartPointer<const VectorType> position;

    SmartPointer<const VectorType> velocity;

    MechanicsUpdateFlags update_flags;

    std::vector<Tensor<2, spacedim>> FF;

    std::vector<Tensor<2, spacedim>> FF_inv_T;

    std::vector<double> det_FF;

    std::vector<Tensor<1, spacedim>> position_values;

    std::vector<Tensor<1, spacedim>> velocity_values;
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
    {
      if (update_flags & update_FF_inv_T)
        update_flags |= update_FF;
      if (update_flags & update_det_FF)
        update_flags |= update_FF;
    }

    // Check some things:
    if (update_flags | update_FF)
      {
        Assert(this->fe_values->get_update_flags() &
                 UpdateFlags::update_gradients,
               ExcMessage("This class needs gradients"));
      }
    if ((update_flags | update_position_values) ||
        (update_flags | update_velocity_values))
      {
        Assert(this->fe_values->get_update_flags() & UpdateFlags::update_values,
               ExcMessage("This class needs gradients"));
      }

    // Set up arrays:
    if (update_flags & MechanicsUpdateFlags::update_FF)
      FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_FF_inv_T)
      FF_inv_T.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_det_FF)
      det_FF.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_position_values)
      position_values.resize(this->fe_values->n_quadrature_points);
    if (update_flags & MechanicsUpdateFlags::update_velocity_values)
      velocity_values.resize(this->fe_values->n_quadrature_points);
  }

  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit()
  {
    const FEValuesExtractors::Vector vec(0);

    if (update_flags & update_FF)
      (*fe_values)[vec].get_function_gradients(*position, FF);

    for (unsigned int q = 0; q < fe_values->n_quadrature_points; ++q)
      {
        if (update_flags & update_FF_inv_T)
          FF_inv_T[q] = transpose(invert(FF[q]));
        if (update_flags & update_det_FF)
          det_FF[q] = determinant(FF[q]);
      }

    if (update_flags & update_position_values)
      (*fe_values)[vec].get_function_values(*position, position_values);

    if (update_flags & update_velocity_values)
      (*fe_values)[vec].get_function_values(*velocity, position_values);
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
} // namespace fdl

#endif
