#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/force_contribution.h>
#include <fiddle/mechanics/mechanics_utilities.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/matrix_free.h>
FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/matrix_free/operators.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <cmath>
#include <fstream>

#include "../tests.h"

// Test compute_volumetric_pk1_load_vector where the force itself is given by a
// finite element field. The errors here are better because we directly use the
// values of the position (and not its derivatives).

using namespace dealii;
using namespace SAMRAI;

template <int spacedim>
class PK1Diagonal : public Function<spacedim>
{
public:
  PK1Diagonal()
    : Function<spacedim>(spacedim)
  {}

  double
  value(const Point<spacedim> &p,
        const unsigned int     component = 0) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());

    const double tau = 2.0 * numbers::PI;
    if (component == 0)
      return -tau * std::sin(tau * p[0]);
    if (component == 1)
      return -tau * std::sin(tau * p[1]);

    return std::numeric_limits<double>::signaling_NaN();
  }
};

template <int spacedim>
class PK1Div : public Function<spacedim>
{
public:
  PK1Div()
    : Function<spacedim>(spacedim)
  {}

  double
  value(const Point<spacedim> &p,
        const unsigned int     component = 0) const override
  {
    Assert(spacedim == 2, fdl::ExcFDLNotImplemented());

    const double tau = 2.0 * numbers::PI;
    if (component == 0)
      return -tau * tau * std::cos(tau * p[0]);
    if (component == 1)
      return -tau * tau * std::cos(tau * p[1]);

    return std::numeric_limits<double>::signaling_NaN();
  }
};

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
    return fdl::MechanicsUpdateFlags::update_position_values;
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
    const FEValuesBase<dim, spacedim> &fe_values = me_values.get_fe_values();
    (void)fe_values;
    Assert(stresses.size() == fe_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());
    Assert(this->get_cell_quadrature().size() ==
             fe_values.get_quadrature_points().size(),
           fdl::ExcFDLInternalError());

    const std::vector<Tensor<1, spacedim>> &positions =
      me_values.get_position_values();
    for (unsigned int qp_n = 0; qp_n < positions.size(); ++qp_n)
      {
        // Using the finite element field here lowers the convergence rate when
        // we don't have a uniform mesh
#if 1
        stresses[qp_n]       = 0.0;
        stresses[qp_n][0][0] = positions[qp_n][0];
        stresses[qp_n][1][1] = positions[qp_n][1];
#else
        const auto &qp = me_values.get_fe_values().quadrature_point(qp_n);
        stresses[qp_n] = 0.0;
        stresses[qp_n][0][0] =
          -2.0 * numbers::PI * std::sin(2 * numbers::PI * qp[0]);
        stresses[qp_n][1][1] =
          -2.0 * numbers::PI * std::sin(2 * numbers::PI * qp[1]);
#endif
      }
  }
};

template <int dim, int spacedim = dim>
void
test(const bool use_simplex)
{
  // setup deal.II stuff:
  const MPI_Comm comm = MPI_COMM_WORLD;
  std::ofstream  output;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      output.open("output", std::ios::app);
      if (use_simplex)
        output << "Using simplices" << std::endl;
      else
        output << "Using hypercubes" << std::endl;
    }
  constexpr int      fe_degree = 2;
  double             old_error = 1.0;
  const unsigned int max_refinements =
    use_simplex ? 8 - fe_degree : 9 - fe_degree;
  for (unsigned int n_refinements = 0; n_refinements < max_refinements;
       ++n_refinements)
    {
      parallel::shared::Triangulation<dim, spacedim> tria(comm);
      if (use_simplex)
        {
          parallel::shared::Triangulation<dim, spacedim> hypercube_tria(comm);
          GridGenerator::hyper_cube(hypercube_tria);
          GridGenerator::convert_hypercube_to_simplex_mesh(hypercube_tria,
                                                           tria);
        }
      else
        {
          GridGenerator::hyper_cube(tria);
        }
      tria.refine_global(n_refinements);
      GridTools::distort_random(0.25, tria);
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        output << "Number of cells = " << tria.n_active_cells() << std::endl;

      std::unique_ptr<FiniteElement<dim, spacedim>> base_fe;
      if (use_simplex)
        base_fe = std::make_unique<FE_SimplexP<dim, spacedim>>(fe_degree);
      else
        base_fe = std::make_unique<FE_Q<dim, spacedim>>(fe_degree);
      FESystem<dim, spacedim>          fe(*base_fe, spacedim);
      std::unique_ptr<Quadrature<dim>> quad_p(
        new QGaussSimplex<dim>(fe.tensor_degree() + 1));
      std::unique_ptr<Quadrature<dim>> quad_q(
        new QGauss<dim>(fe.tensor_degree() + 1));
      const Quadrature<dim> &quadrature = use_simplex ? *quad_p : *quad_q;
      std::unique_ptr<Mapping<dim, spacedim>> mapping_p(
        new MappingFE<dim, spacedim>(FE_SimplexP<dim, spacedim>(1)));
      std::unique_ptr<Mapping<dim, spacedim>> mapping_q(
        new MappingQ1<dim, spacedim>());
      const Mapping<dim, spacedim> &mapping =
        use_simplex ? *mapping_p : *mapping_q;

      DoFHandler<dim, spacedim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      IndexSet locally_owned_dofs, locally_relevant_dofs;
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      auto partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                      locally_relevant_dofs,
                                                      comm);

      MatrixFreeOperators::MassOperator<dim, -1, 0, dim> mass_operator;
      auto matrix_free = std::make_shared<MatrixFree<dim, double>>();
      AffineConstraints<double> constraints;
      constraints.close();
      matrix_free->reinit(mapping, dof_handler, constraints, quadrature);

      mass_operator.initialize(matrix_free);
      mass_operator.compute_diagonal();
      // TODO - PreconditionJacobi doesn't work with FE_SimplexP(2)
      // PreconditionJacobi<decltype(mass_operator)> mass_preconditioner;
      PreconditionIdentity mass_preconditioner;
      // mass_preconditioner.initialize(mass_operator, 1.0);

      // and the test itself:
      {
        // we are integrating a trig function so be generous with quadrature
        std::unique_ptr<Quadrature<dim>> quad_p2(
          new QWitherdenVincentSimplex<dim>(fe.tensor_degree() + 2));
        std::unique_ptr<Quadrature<dim>> quad_q2(
          new QGauss<dim>(fe.tensor_degree() + 2));
        const Quadrature<dim> &quadrature2 = use_simplex ? *quad_p2 : *quad_q2;
        StressContribution<dim, spacedim> s1(quadrature2);

        std::vector<fdl::ForceContribution<dim, spacedim> *> stress_ptrs{&s1};
        // This test does read the position
        LinearAlgebra::distributed::Vector<double> current_position(
          partitioner),
          current_velocity(partitioner), force_rhs(partitioner),
          force(partitioner);
        VectorTools::interpolate(dof_handler,
                                 PK1Diagonal<spacedim>(),
                                 current_position);
        current_position.update_ghost_values();

#if 1
        fdl::compute_volumetric_pk1_load_vector(dof_handler,
                                                mapping,
                                                stress_ptrs,
                                                {},
                                                0.0,
                                                current_position,
                                                current_velocity,
                                                force_rhs);
#else
        VectorTools::create_right_hand_side(
          mapping, dof_handler, quadrature2, PK1Div<spacedim>(), force_rhs);
#endif
        force_rhs.compress(VectorOperation::add);
        // Do the L2 projection:
        SolverControl                 control(1000, 1e-8 * force_rhs.l1_norm());
        SolverCG<decltype(force_rhs)> cg(control);
        cg.solve(mass_operator, force, force_rhs, mass_preconditioner);
        force.update_ghost_values();

        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(force, "F");
          data_out.add_data_vector(current_position, "X");

          data_out.build_patches();
          const std::string name =
            use_simplex ? "solution-simplex" : "solution-hypercube";
          data_out.write_vtu_with_pvtu_record(
            "./", name, n_refinements, comm, 8);
        }

        Vector<double> cell_error(tria.n_active_cells());
        VectorTools::integrate_difference(mapping,
                                          dof_handler,
                                          force,
                                          PK1Div<spacedim>(),
                                          cell_error,
                                          quadrature2,
                                          VectorTools::L2_norm);

        const double global_error =
          compute_global_error(tria, cell_error, VectorTools::L2_norm);

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            output << "global error = " << global_error << std::endl;
            output << "error ratio  = " << old_error / global_error
                   << std::endl;
          }
        old_error = global_error;
      }
    }
}

int
main(int argc, char **argv)
{
  // Best way to empty the file
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::ofstream("output");
  test<2>(/*use_simplex = */ false);
  test<2>(/*use_simplex = */ true);
}
