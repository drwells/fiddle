#ifndef included_fiddle_mechanics_active_strain_h
#define included_fiddle_mechanics_active_strain_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <algorithm>

namespace fdl
{
  using namespace dealii;

  /**
   * Interface class for defining active strains on Parts.
   *
   * ActiveStrain objects work together with MechanicsValues to modify the
   * deformation gradients specified to ForceContribution objects. These
   * modification operations are definde by push_deformation_gradient_forward()
   * and pull_stress_back().
   */
  template <int dim, int spacedim = dim, typename Number = double>
  class ActiveStrain
  {
  public:
    /**
     * Constructor.
     */
    ActiveStrain(const std::vector<types::material_id> &material_ids);

    /**
     * Default virtual destructor.
     */
    virtual ~ActiveStrain() = default;

    /**
     * Do any time-dependent initialization of the strain.
     */
    virtual void
    setup_strain(const double                                      time,
                 const LinearAlgebra::distributed::Vector<double> &position,
                 const LinearAlgebra::distributed::Vector<double> &velocity);

    /**
     * Matching function to setup_strain(), which may deallocate memory or
     * perform other cleanup actions.
     */
    virtual void
    finish_strain(const double time);

    /**
     * Push the deformation gradients forward by the formula
     *
     *     FF_E = FF FF_A^-1
     *
     * in which FF is the deformation gradient and FF_A is the active stress
     * tensor defined by this class.
     */
    virtual void
    push_deformation_gradient_forward(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      const ArrayView<Tensor<2, spacedim, Number>>                      &FF,
      ArrayView<Tensor<2, spacedim, Number>> &push_forward_FF) const = 0;

    /**
     * Pull the first Piola-Kirchoff stress tensors back by the formula
     *
     *     PP = det(FF_A) PP_E(FF_E) FF_A^-T
     *
     * in which PP_E is the first Piola-Kirchoff stress tensor (which is
     * computed with FF_E) and FF_A is the active stress tensor defined by
     * this class.
     */
    virtual void
    pull_stress_back(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      const ArrayView<Tensor<2, spacedim, Number>> &push_forward_stress,
      ArrayView<Tensor<2, spacedim, Number>>       &stress) const = 0;

    /**
     * Return the material ids over which the present object is defined.
     */
    const std::vector<types::material_id> &
    get_material_ids() const;

  private:
    std::vector<types::material_id> material_ids;
  };

  // --------------------------- inline functions --------------------------- //

  template <int dim, int spacedim, typename Number>
  ActiveStrain<dim, spacedim, Number>::ActiveStrain(
    const std::vector<types::material_id> &material_ids)
    : material_ids(material_ids)
  {
    std::sort(this->material_ids.begin(), this->material_ids.end());
    this->material_ids.erase(std::unique(this->material_ids.begin(),
                                         this->material_ids.end()),
                             this->material_ids.end());
  }

  template <int dim, int spacedim, typename Number>
  void
  ActiveStrain<dim, spacedim, Number>::setup_strain(
    const double                                      time,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
  {
    (void)time;
    (void)position;
    (void)velocity;
  }

  template <int dim, int spacedim, typename Number>
  void
  ActiveStrain<dim, spacedim, Number>::finish_strain(const double time)
  {
    (void)time;
  }

  template <int dim, int spacedim, typename Number>
  const std::vector<types::material_id> &
  ActiveStrain<dim, spacedim, Number>::get_material_ids() const
  {
    return material_ids;
  }

} // namespace fdl

#endif
