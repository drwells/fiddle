#ifndef included_fiddle_mechanics_utilities_h
#define included_fiddle_mechanics_utilities_h

#include <fiddle/mechanics/force_contribution.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Compute the volumetric component of the PK1 stress and add it into the
   * given load vector.
   */
  template <int dim, int spacedim = dim>
  void
  compute_volumetric_pk1_load_vector(
    const DoFHandler<dim, spacedim> &                     dof_handler,
    const Mapping<dim, spacedim> &                        mapping,
    const std::vector<ForceContribution<dim, spacedim> *> stress_contributions,
    const LinearAlgebra::distributed::Vector<double> &    current_position,
    const LinearAlgebra::distributed::Vector<double> &    current_velocity,
    LinearAlgebra::distributed::Vector<double> &          force_rhs);

} // namespace fdl

#endif
