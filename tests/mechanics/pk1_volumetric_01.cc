#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <fstream>

#include "../tests.h"

// Test compute_volumetric_pk1_load_vector

using namespace dealii;
using namespace SAMRAI;


template <int dim, int spacedim = dim>
class StressContribution : public fdl::ForceContribution<dim, spacedim>
{
public:
  StressContribution(const Quadrature<dim> &quad)
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
    ArrayView<Tensor<2, spacedim, double>> &   stresses) const override
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

template <int spacedim>
class PK1Div : public Function<spacedim>
{
public:
  PK1Div(const double multiplier)
    : Function<spacedim>(spacedim)
    , multiplier(multiplier)
  {}

  double
  value(const Point<spacedim> &p,
        const unsigned int     component = 0) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());

    const double tau = 2.0 * numbers::PI;
    const double pi  = numbers::PI;
    if (component == 0)
      return multiplier *
             (tau * std::cos(tau * p[0]) +
              std::cos(2.0 * tau * p[0]) * pi * std::cos(pi * p[1]));
    if (component == 1)
      return multiplier * (tau * std::cos(tau * p[1]));

    return std::numeric_limits<double>::signaling_NaN();
  }

private:
  double multiplier;
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
  for (unsigned int n_refinements = 0; n_refinements < 7; ++n_refinements)
    {
      parallel::shared::Triangulation<dim, spacedim> tria(comm);
      GridGenerator::hyper_cube(tria);
      tria.refine_global(n_refinements);
      FESystem<dim, spacedim>   fe(FE_Q<dim, spacedim>(2), spacedim);
      DoFHandler<dim, spacedim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);
      MappingCartesian<dim, spacedim> mapping;
      QGauss<dim>                     quadrature(fe.degree + 1);

      IndexSet locally_owned_dofs, locally_relevant_dofs;
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      auto partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                      locally_relevant_dofs,
                                                      comm);

      MatrixFreeOperators::MassOperator<dim, 2, 2 + 1, dim> mass_operator;
      auto matrix_free = std::make_shared<MatrixFree<dim, double>>();
      AffineConstraints<double> constraints;
      constraints.close();
      matrix_free->reinit(mapping, dof_handler, constraints, quadrature);

      mass_operator.initialize(matrix_free);
      mass_operator.compute_diagonal();
      PreconditionJacobi<decltype(mass_operator)> mass_preconditioner;
      mass_preconditioner.initialize(mass_operator, 1.0);

      // and the test itself:
      {
        // Make sure we can handle multiple quadratures
        QGauss<dim>                       quadrature2(fe.degree + 1);
        QGauss<dim>                       quadrature3(fe.degree + 2);
        StressContribution<dim, spacedim> s1(quadrature2);
        StressContribution<dim, spacedim> s2(quadrature3);
        StressContribution<dim, spacedim> s3(quadrature2);
        StressContribution<dim, spacedim> s4(quadrature2);

        std::vector<fdl::ForceContribution<dim, spacedim> *> stress_ptrs{&s1,
                                                                         &s2,
                                                                         &s3,
                                                                         &s4};
        // This test doesn't read the position or velocity
        LinearAlgebra::distributed::Vector<double> current_position(
          partitioner),
          current_velocity(partitioner), force_rhs(partitioner),
          force(partitioner);

        fdl::compute_volumetric_pk1_load_vector(dof_handler,
                                                mapping,
                                                stress_ptrs,
                                                0.0,
                                                current_position,
                                                current_velocity,
                                                force_rhs);
        force_rhs.compress(VectorOperation::add);
        // Do the L2 projection:
        SolverControl control(1000, 1e-14 * force_rhs.l2_norm());
        SolverCG<decltype(force_rhs)> cg(control);
        cg.solve(mass_operator, force, force_rhs, mass_preconditioner);
        force.update_ghost_values();

        Vector<double> cell_error(tria.n_active_cells());
        VectorTools::integrate_difference(mapping,
                                          dof_handler,
                                          force,
                                          PK1Div<spacedim>(stress_ptrs.size()),
                                          cell_error,
                                          quadrature3,
                                          VectorTools::L2_norm);

        const double global_error =
          compute_global_error(tria, cell_error, VectorTools::L2_norm);

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          output << "global error = " << global_error << '\n';
      }
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);
  test<2>();
}
