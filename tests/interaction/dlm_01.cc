#include <fiddle/interaction/dlm_method.h>

#include <fiddle/mechanics/force_contribution_lib.h>
#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <fstream>
#include <vector>

// basic test for DLM stuff

using namespace dealii;

// Simple DLM class: a structure moving with velocity (1, 1, 1).
template <int dim, int spacedim = dim>
class DLMMethod : public fdl::DLMMethodBase<dim, spacedim>
{
public:
  DLMMethod(const LinearAlgebra::distributed::Vector<double> &position)
    : reference_position(position)
  {}

  virtual void
  get_mechanics_position(
    const double                                time,
    LinearAlgebra::distributed::Vector<double> &position) const override
  {
    position = reference_position;
    for (std::size_t i = 0; i < position.locally_owned_size(); ++i)
      position.local_element(i) += time;

    position.update_ghost_values();
  }

  virtual const LinearAlgebra::distributed::Vector<double> &
  get_current_mechanics_position() const override
  {
    return reference_position;
  }

protected:
  LinearAlgebra::distributed::Vector<double> reference_position;
};

int
main()
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2>   fe(FE_Q<2>(1), 2);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const MPI_Comm comm = MPI_COMM_WORLD;

  LinearAlgebra::distributed::Vector<double> reference(
    dof_handler.locally_owned_dofs(), comm);
  VectorTools::interpolate(dof_handler,
                           Functions::IdentityFunction<2>(),
                           reference);
  DLMMethod<2> dlm_method(reference);

  // We have to do some manual setup of stuff normally done inside the library
  QMidpoint<2>            quadrature;
  FEValues<2>             fe_values(fe, quadrature, update_values);
  fdl::MechanicsValues<2> m_values(fe_values,
                                   reference,
                                   reference,
                                   fdl::MechanicsUpdateFlags::update_nothing);

  const double     spring_constant = 10.0;
  fdl::DLMForce<2> dlm_force(quadrature,
                             spring_constant,
                             dof_handler,
                             dlm_method);

  std::vector<Tensor<1, 2>> forces(quadrature.size());
  std::ofstream             output("output");
  for (unsigned int i = 0; i < 3; ++i)
    {
      const double time = i * 0.5;
      output << "test " << i << ": t = " << time
             << ", spring constant = " << spring_constant << '\n';
      dlm_force.setup_force(time, reference, reference);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          // more manual stuff that normally would not be necessary
          fe_values.reinit(cell);
          output << "cell center = " << cell->center() << '\n';
          auto view = make_array_view(forces);
          dlm_force.compute_volume_force(0.0, m_values, cell, view);
          output << "forces = ";
          for (const Tensor<1, 2> &f : forces)
            output << f << '\n';
        }
      dlm_force.finish_force(time);
    }
}
