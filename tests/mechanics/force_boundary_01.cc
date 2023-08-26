#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>
FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/matrix_free/operators.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <fstream>

#include "../tests.h"

// Test compute_boundary_force_load_vector() by setting up the boundary force as
// part of a Laplace solve.

using namespace dealii;
using namespace SAMRAI;


bool integrated_face = false;

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);
    if (component == 0)
      return 4.0 * std::sin(p[0]) * std::cos(p[1]);
    return 4.0 * std::cos(p[0]) * std::sin(p[1]);
  }
};



template <int dim>
class Forcing : public Function<dim>
{
public:
  Forcing()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);
    if (component == 0)
      return 8.0 * std::sin(p[0]) * std::cos(p[1]);
    return 8.0 * std::cos(p[0]) * std::sin(p[1]);
  }
};

// 'force' used for the Neumann boundary condition
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
  compute_boundary_force(
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
    const auto                        &face_values =
      dynamic_cast<const FEFaceValues<dim, spacedim> &>(fe_values);
    Assert(this->get_face_quadrature().size() ==
             face_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());

    for (unsigned int qp_n : face_values.quadrature_point_indices())
      {
        const auto                 &p = fe_values.quadrature_point(qp_n);
        Tensor<1, spacedim, double> ug0, ug1;
        ug0[0] = 4.0 * std::cos(p[0]) * std::cos(p[1]);
        ug0[1] = -4.0 * std::sin(p[0]) * std::sin(p[1]);
        ug1[0] = -4.0 * std::sin(p[0]) * std::sin(p[1]);
        ug1[1] = 4.0 * std::cos(p[0]) * std::cos(p[1]);
        // we use two copies of this boundary force in this test so this is
        // halved
        const auto &N   = face_values.normal_vector(qp_n);
        forces[qp_n][0] = ug0 * N * 0.5;
        forces[qp_n][1] = ug1 * N * 0.5;
      }
    integrated_face = true;
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
  double old_error = 1.0;
  for (unsigned int n_refinements = 0; n_refinements < 6; ++n_refinements)
    {
      const auto mesh_partitioner =
        parallel::shared::Triangulation<dim,
                                        spacedim>::Settings::partition_zorder;
      parallel::shared::Triangulation<dim, spacedim> tria(comm,
                                                          {},
                                                          false,
                                                          mesh_partitioner);
      GridGenerator::hyper_shell(tria, Point<dim>(), 1.0, 2.0, 0, true);
      tria.refine_global(n_refinements);
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        output << "Number of cells = " << tria.n_active_cells() << std::endl;
      constexpr int             degree = 1;
      FESystem<dim, spacedim>   fe(FE_Q<dim, spacedim>(degree), spacedim);
      DoFHandler<dim, spacedim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);
      MappingQ1<dim, spacedim> mapping;
      QGauss<dim>              quadrature(degree + 1);

      IndexSet locally_owned_dofs, locally_relevant_dofs;
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      auto partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                      locally_relevant_dofs,
                                                      comm);
      // inner boundary has id 0 - set to Dirichlet BCs
      AffineConstraints<double> constraints;
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, ExactSolution<dim>(), constraints);
      constraints.close();

      AffineConstraints<double> no_constraints;
      no_constraints.close();
      auto matrix_free = std::make_shared<MatrixFree<dim, double>>();
      matrix_free->reinit(mapping, dof_handler, no_constraints, quadrature);

      MatrixFreeOperators::LaplaceOperator<dim, degree, degree + 1, spacedim>
        laplace_operator;
      laplace_operator.initialize(matrix_free);
      laplace_operator.compute_diagonal();
      PreconditionJacobi<decltype(laplace_operator)> laplace_preconditioner;
      laplace_preconditioner.initialize(laplace_operator, 1.0);

      // and the test itself:
      {
        // Make sure we can handle multiple quadratures
        QGauss<dim - 1>              quadrature2(degree + 1);
        QGauss<dim - 1>              quadrature3(degree + 2);
        BoundaryForce<dim, spacedim> s1(quadrature2);
        BoundaryForce<dim, spacedim> s2(quadrature3);

        std::vector<fdl::ForceContribution<dim, spacedim> *> force_ptrs{&s1,
                                                                        &s2};
        // This test doesn't read the position or velocity
        LinearAlgebra::distributed::Vector<double> current_position(
          partitioner),
          current_velocity(partitioner), force_rhs(partitioner),
          force(partitioner), inhomogeneity(partitioner);

        VectorTools::create_right_hand_side(
          mapping, dof_handler, quadrature, Forcing<dim>(), force_rhs);

        // The boundary force doesn't touch constrained DoFs so we can just add
        // it to the vector without worrying about constraints
        fdl::compute_boundary_force_load_vector(dof_handler,
                                                mapping,
                                                force_ptrs,
                                                0.0,
                                                current_position,
                                                current_velocity,
                                                force_rhs);
        force_rhs.compress(VectorOperation::add);

        // Do the L2 projection:
        SolverControl control(1000, 1e-10 * force_rhs.l2_norm());
        SolverCG<decltype(force_rhs)> cg(control);

        // For simplicity we implement constraints outside the operator itself
        // and use LinearOperator to apply them on the fly
        const auto op_a =
          linear_operator<decltype(force_rhs)>(laplace_operator);
        const auto op_amod = constrained_linear_operator(constraints, op_a);
        decltype(force_rhs) rhs_mod =
          constrained_right_hand_side(constraints, op_a, force_rhs);
        cg.solve(op_amod, force, rhs_mod, laplace_preconditioner);
        constraints.distribute(force);
        force.update_ghost_values();

        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(force, "F");

          data_out.build_patches();
          const std::string name = "solution-hypercube";
          data_out.write_vtu_with_pvtu_record(
            "./", name, n_refinements, comm, 8);
        }

        Vector<double> cell_error(tria.n_active_cells());
        VectorTools::integrate_difference(mapping,
                                          dof_handler,
                                          force,
                                          ExactSolution<dim>(),
                                          cell_error,
                                          quadrature,
                                          VectorTools::L2_norm);

        const double global_error =
          compute_global_error(tria, cell_error, VectorTools::L2_norm);

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            output << "global error = " << global_error << '\n';
            output << "error ratio  = " << old_error / global_error
                   << std::endl;
          }
        old_error = global_error;
      }
    }

  AssertThrow(integrated_face,
              ExcMessage("we should have integrated over a face"));
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);
  test<2>();
}
