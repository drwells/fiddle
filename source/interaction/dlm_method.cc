#include <fiddle/interaction/dlm_method.h>

namespace fdl
{
  template <int dim, int spacedim>
  DLMForce<dim, spacedim>::DLMForce(
    const Quadrature<dim> &          quad,
    const double                     spring_constant,
    const DoFHandler<dim, spacedim> &dof_handler,
    DLMMethodBase<dim, spacedim> &   dlm)
    : SpringForce<dim, spacedim>(quad,
                                 spring_constant,
                                 dof_handler,
                                 dlm.get_position())
    , dlm(&dlm)
  {}

  template <int dim, int spacedim>
  void
  DLMForce<dim, spacedim>::setup_force(
    const double                                      time,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> & /*velocity*/)
  {
    this->current_position = &position;
    dlm->update_external_position(time, position);
    dlm->get_position(time, this->reference_position);
  }



  template class DLMForce<NDIM - 1, NDIM>;
  template class DLMForce<NDIM, NDIM>;
} // namespace fdl
