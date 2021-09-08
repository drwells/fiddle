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
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_rhs.h>

#include <cmath>
#include <fstream>

#include "../tests.h"

// Test compute_volumetric_force_load_vector. This is essentially just an L2
// projection between two finite element fields.

using namespace dealii;
using namespace SAMRAI;


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
void
test(const bool use_simplex)
{
  // setup deal.II stuff:
  const MPI_Comm comm = MPI_COMM_WORLD;
  std::ofstream  output;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      output.open("output");
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
      FESystem<dim, spacedim>   fe(*base_fe, spacedim);
      DoFHandler<dim, spacedim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);
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
        std::unique_ptr<Quadrature<dim>> quad_p2(
          new QWitherdenVincentSimplex<dim>(fe.tensor_degree() + 2));
        std::unique_ptr<Quadrature<dim>> quad_q2(
          new QGauss<dim>(fe.tensor_degree() + 2));
        const Quadrature<dim> &quadrature2 = use_simplex ? *quad_p2 : *quad_q2;
        Force<dim, spacedim>   s1(quadrature2);
        Force<dim, spacedim>   s2(quadrature);
        Force<dim, spacedim>   s3(quadrature2);
        Force<dim, spacedim>   s4(quadrature2);

        std::vector<fdl::ForceContribution<dim, spacedim> *> force_ptrs{
          &s1, &s2, &s3, &s4};
        // This test does read the position
        LinearAlgebra::distributed::Vector<double> current_position(
          partitioner),
          current_velocity(partitioner), force_rhs(partitioner),
          force(partitioner);
        VectorTools::interpolate(dof_handler,
                                 Position<spacedim>(),
                                 current_position);
        current_position.update_ghost_values();

        fdl::compute_volumetric_force_load_vector(dof_handler,
                                                  mapping,
                                                  force_ptrs,
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
                                          ExactSolution<dim>(),
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
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);
  test<2>(/*use_simplex = */ false);
  test<2>(/*use_simplex = */ true);
}
