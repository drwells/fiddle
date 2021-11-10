#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <fstream>

#include "../tests.h"

// Test compute_load_vector() - originally this function was split into three
// parts, this test examines the whole thing at once by printing values

using namespace dealii;
using namespace SAMRAI;

template <int dim>
class Position : public Function<dim>
{
public:
  Position()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);
    if (component == 0)
      return std::sin(p[0]) * std::cos(p[1]);
    return std::cos(p[0]) * std::sin(p[1]);
  }
};



template <int dim, int spacedim = dim>
class Stress : public fdl::ForceContribution<dim, spacedim>
{
public:
  Stress(const Quadrature<dim> &quad)
    : fdl::ForceContribution<dim, spacedim>(quad)
  {}

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::MechanicsUpdateFlags::update_nothing;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_quadrature_points;
  }

  virtual bool
  is_stress() const override
  {
    return true;
  }

  virtual void
  compute_stress(
    const double /*time*/,
    const fdl::MechanicsValues<dim, spacedim> &me_values,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      & /*cell*/,
    ArrayView<Tensor<2, spacedim, double>> &stresses) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());
    // We have to chose a function here such that FF * n = (0, 0) on the
    // boundaries so that the surface integral goes away.

    const FEValuesBase<dim, spacedim> &fe_values = me_values.get_fe_values();
    Assert(stresses.size() == fe_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());
    Assert(this->get_cell_quadrature().size() ==
             fe_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());

    const double tau = 2.0 * numbers::PI;
    const double pi  = numbers::PI;

    for (unsigned int qp_n : fe_values.quadrature_point_indices())
      {
        const auto &                 p  = fe_values.quadrature_point(qp_n);
        Tensor<2, spacedim, double> &PP = stresses[qp_n];
        // must be 0 when x = 0 or x = 1
        PP[0][0] = std::sin(tau * p[0]);
        // must be 0 when y = 0 or y = 1
        PP[0][1] = std::cos(4.0 * pi * p[0]) * std::sin(pi * p[1]);
        // must be 0 when x = 0 or x = 1
        PP[1][0] = 0;
        // must be 0 when y = 0 or y = 1
        PP[1][1] = std::sin(tau * p[1]);
      }
  }
};

template <int dim, int spacedim = dim>
class Force : public fdl::ForceContribution<dim, spacedim>
{
public:
  Force(const Quadrature<dim> &quad)
    : fdl::ForceContribution<dim, spacedim>(quad)
  {}

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_default;
  }

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::MechanicsUpdateFlags::update_nothing |
           fdl::MechanicsUpdateFlags::update_position_values;
  }

  virtual bool
  is_volume_force() const override
  {
    return true;
  }

  virtual void
  compute_force(const double /*time*/,
                const fdl::MechanicsValues<dim, spacedim> &me_values,
                ArrayView<Tensor<1, spacedim, double>> &forces) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());

    Assert(forces.size() == this->get_cell_quadrature().size(),
           fdl::ExcFDLInternalError());
    const std::vector<Tensor<1, spacedim>> &positions =
      me_values.get_position_values();
    Assert(this->get_cell_quadrature().size() == positions.size(),
           fdl::ExcFDLInternalError());

    std::copy(positions.begin(), positions.end(), forces.begin());
  }
};

template <int dim, int spacedim = dim>
class BoundaryForce : public fdl::ForceContribution<dim, spacedim>
{
public:
  BoundaryForce(const Quadrature<dim - 1> &quad)
    : fdl::ForceContribution<dim, spacedim>(quad)
  {}

  virtual fdl::MechanicsUpdateFlags
  get_mechanics_update_flags() const override
  {
    return fdl::MechanicsUpdateFlags::update_nothing;
  }

  virtual UpdateFlags
  get_update_flags() const override
  {
    return UpdateFlags::update_quadrature_points |
           UpdateFlags::update_normal_vectors;
  }

  virtual bool
  is_boundary_force() const override
  {
    return true;
  }

  virtual void
  compute_surface_force(
    const double /*time*/,
    const fdl::MechanicsValues<dim, spacedim> &me_values,
    const typename Triangulation<dim, spacedim>::active_face_iterator &face,
    ArrayView<Tensor<1, spacedim, double>> &forces) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());
    Assert(face->at_boundary(), fdl::ExcFDLInternalError());

    // boundary ids are hard-coded in this test
    if (face->boundary_id() != 1)
      {
        std::fill(forces.begin(), forces.end(), Tensor<1, spacedim, double>());
        return;
      }

    const FEValuesBase<dim, spacedim> &fe_values = me_values.get_fe_values();
    const auto &                       face_values =
      dynamic_cast<const FEFaceValues<dim, spacedim> &>(fe_values);
    Assert(this->get_face_quadrature().size() ==
             face_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());

    for (unsigned int qp_n : face_values.quadrature_point_indices())
      {
        const auto &                p = fe_values.quadrature_point(qp_n);
        Tensor<1, spacedim, double> ug0, ug1;
        ug0[0] = 4.0 * std::cos(p[0]) * std::cos(p[1]);
        ug0[1] = -4.0 * std::sin(p[0]) * std::sin(p[1]);
        ug1[0] = -4.0 * std::sin(p[0]) * std::sin(p[1]);
        ug1[1] = 4.0 * std::cos(p[0]) * std::cos(p[1]);
        // we use two copies of this surface force in this test so this is
        // halved
        const auto &N   = face_values.normal_vector(qp_n);
        forces[qp_n][0] = ug0 * N * 0.5;
        forces[qp_n][1] = ug1 * N * 0.5;
      }
  }
};



template <int dim, int spacedim = dim>
void
test()
{
  // setup deal.II stuff:
  const MPI_Comm comm = MPI_COMM_WORLD;
  std::ofstream  output;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    output.open("output");

  parallel::shared::Triangulation<dim, spacedim> tria(comm);
  GridGenerator::hyper_shell(tria, Point<dim>(), 1.0, 2.0, 0, true);
  tria.refine_global(1);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    output << "Number of cells = " << tria.n_active_cells() << std::endl;
  constexpr int             degree = 1;
  FESystem<dim, spacedim>   fe(FE_Q<dim, spacedim>(degree), spacedim);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQ<dim, spacedim> mapping(1);
  QGauss<dim>             quadrature1(degree + 1);

  IndexSet locally_owned_dofs, locally_relevant_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  auto partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                  locally_relevant_dofs,
                                                  comm);

  // and the test itself:
  {
    // Make sure we can handle multiple quadratures
    QGauss<dim - 1>              face_quadrature2(degree + 1);
    QGauss<dim - 1>              face_quadrature3(degree + 2);
    QGauss<dim>                  quadrature2(degree + 1);
    QGauss<dim>                  quadrature3(degree + 2);
    BoundaryForce<dim, spacedim> b1(face_quadrature2);
    BoundaryForce<dim, spacedim> b2(face_quadrature3);
    Force<dim, spacedim>         f1(quadrature2);
    Force<dim, spacedim>         f2(quadrature3);
    Stress<dim, spacedim>        s1(quadrature2);
    Stress<dim, spacedim>        s2(quadrature3);
    Stress<dim, spacedim>        s3(quadrature1);

    std::vector<fdl::ForceContribution<dim, spacedim> *> force_ptrs{
      &b1, &f1, &s1, &b2, &f2, &s2, &s3};
    // This test does read the position
    LinearAlgebra::distributed::Vector<double> current_position(partitioner),
      current_velocity(partitioner), force_rhs(partitioner), force(partitioner);
    VectorTools::interpolate(dof_handler,
                             Position<spacedim>(),
                             current_position);
    current_position.update_ghost_values();

    fdl::compute_load_vector(dof_handler,
                             mapping,
                             force_ptrs,
                             0.0,
                             current_position,
                             current_velocity,
                             force_rhs);
    force_rhs.compress(VectorOperation::add);

    std::ostringstream local_out;
    force_rhs.print(local_out, 16, true, false);

    print_strings_on_0(local_out.str(), comm, output);
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);
  test<2>();
}
