#ifndef included_fiddle_mechanics_utilities_h
#define included_fiddle_mechanics_utilities_h

#include <fiddle/base/config.h>

#include <fiddle/mechanics/active_strain.h>
#include <fiddle/mechanics/force_contribution.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <vector>

// forward declarations
namespace dealii
{
  template <int, int>
  class DoFHandler;
  template <int, int>
  class Mapping;
} // namespace dealii

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
    const DoFHandler<dim, spacedim>                       &dof_handler,
    const Mapping<dim, spacedim>                          &mapping,
    const std::vector<ForceContribution<dim, spacedim> *> &stress_contributions,
    const std::vector<ActiveStrain<dim, spacedim> *>      &active_strains,
    const double                                           time,
    const LinearAlgebra::distributed::Vector<double>      &current_position,
    const LinearAlgebra::distributed::Vector<double>      &current_velocity,
    LinearAlgebra::distributed::Vector<double>            &force_rhs);

  /**
   * Compute the contribution of volumetric forces and add them to the given
   * load vector.
   */
  template <int dim, int spacedim = dim>
  void
  compute_volumetric_force_load_vector(
    const DoFHandler<dim, spacedim>                       &dof_handler,
    const Mapping<dim, spacedim>                          &mapping,
    const std::vector<ForceContribution<dim, spacedim> *> &force_contributions,
    const double                                           time,
    const LinearAlgebra::distributed::Vector<double>      &current_position,
    const LinearAlgebra::distributed::Vector<double>      &current_velocity,
    LinearAlgebra::distributed::Vector<double>            &force_rhs);

  /**
   * Compute the contribution of boundary forces and add them to the given load
   * vector.
   */
  template <int dim, int spacedim = dim>
  void
  compute_boundary_force_load_vector(
    const DoFHandler<dim, spacedim>                       &dof_handler,
    const Mapping<dim, spacedim>                          &mapping,
    const std::vector<ForceContribution<dim, spacedim> *> &force_contributions,
    const double                                           time,
    const LinearAlgebra::distributed::Vector<double>      &current_position,
    const LinearAlgebra::distributed::Vector<double>      &current_velocity,
    LinearAlgebra::distributed::Vector<double>            &force_rhs);

  /**
   * Combined function that calls all of the previous functions.
   */
  template <int dim, int spacedim = dim>
  void
  compute_load_vector(
    const DoFHandler<dim, spacedim>                       &dof_handler,
    const Mapping<dim, spacedim>                          &mapping,
    const std::vector<ForceContribution<dim, spacedim> *> &force_contributions,
    const std::vector<ActiveStrain<dim, spacedim> *>      &active_strains,
    const double                                           time,
    const LinearAlgebra::distributed::Vector<double>      &current_position,
    const LinearAlgebra::distributed::Vector<double>      &current_velocity,
    LinearAlgebra::distributed::Vector<double>            &force_rhs);
} // namespace fdl

#endif
