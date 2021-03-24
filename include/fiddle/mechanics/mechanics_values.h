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

  /**
   * Class that computes mechanics values.
   *
   * @note This is very much WIP - the minimum viable implementation just needs
   * FF, FF^-T, and det(FF) so that's unconditionally computed for now. In the
   * future I'll add some update flags equivalent that turns these (and useful
   * invariants) on and off.
   */
  template <int dim,
            int spacedim        = dim,
            typename VectorType = LinearAlgebra::distributed::Vector<double>>
  class MechanicsValues
  {
  public:
    MechanicsValues(const FEValuesBase<dim, spacedim> &fe_values,
                    const VectorType &                 X);

    void
    reinit();

    const std::vector<Tensor<2, spacedim>> &
    get_FF() const;

    const std::vector<Tensor<2, spacedim>> &
    get_FF_inv_T() const;

    const std::vector<double> &
    get_det_FF() const;

  protected:
    SmartPointer<const FEValuesBase<dim, spacedim>> fe_values;

    SmartPointer<const VectorType> X;

    std::vector<Tensor<2, spacedim>> FF;

    std::vector<Tensor<2, spacedim>> FF_inv_T;

    std::vector<double> det_FF;
  };

  // Constructor and reinitialization

  template <int dim, int spacedim, typename VectorType>
  MechanicsValues<dim, spacedim, VectorType>::MechanicsValues(
    const FEValuesBase<dim, spacedim> &fe_values,
    const VectorType &                 X)
    : fe_values(&fe_values)
    , X(&X)
  {
    FF.resize(this->fe_values->n_quadrature_points);
    FF_inv_T.resize(this->fe_values->n_quadrature_points);
    det_FF.resize(this->fe_values->n_quadrature_points);

    // like everything else, this needs to be generalized
    Assert(this->fe_values->get_update_flags() & UpdateFlags::update_gradients,
           ExcMessage("This class needs gradients"));
  }

  template <int dim, int spacedim, typename VectorType>
  void
  MechanicsValues<dim, spacedim, VectorType>::reinit()
  {
    const FEValuesExtractors::Vector vec(0);

    (*fe_values)[vec].get_function_gradients(*X, FF);

    for (unsigned int q = 0; q < fe_values->n_quadrature_points; ++q)
      {
        FF_inv_T[q] = transpose(invert(FF[q]));
        det_FF[q]   = determinant(FF[q]);
      }
  }

  // Access functions

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_FF() const
  {
    return FF;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<Tensor<2, spacedim>> &
  MechanicsValues<dim, spacedim, VectorType>::get_FF_inv_T() const
  {
    return FF_inv_T;
  }

  template <int dim, int spacedim, typename VectorType>
  inline const std::vector<double> &
  MechanicsValues<dim, spacedim, VectorType>::get_det_FF() const
  {
    return det_FF;
  }
} // namespace fdl

#endif
