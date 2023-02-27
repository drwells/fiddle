#include <fiddle/mechanics/force_contribution_lib.h>
#include <fiddle/mechanics/mechanics_values.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <deal.II/physics/transformations.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>
#include <memory>
#include <vector>

// Basic tests for forces defined on the boundary only

using namespace dealii;

template <int dim, int spacedim = dim, typename Number = double>
void
print_force(fdl::ForceContribution<dim, spacedim, Number> &   boundary_force,
            const DoFHandler<dim, spacedim> &                 dof_handler,
            const double                                      time,
            const LinearAlgebra::distributed::Vector<double> &position,
            const LinearAlgebra::distributed::Vector<double> &velocity,
            std::ostream &                                    output)
{
  boundary_force.setup_force(time, position, velocity);

  // We have to do some manual setup of stuff normally done inside the library
  const fdl::MechanicsUpdateFlags face_me_flags =
    boundary_force.get_mechanics_update_flags();
  MappingCartesian<dim, spacedim> mapping;
  FEFaceValues<dim, spacedim>     fe_face_values(
    mapping,
    dof_handler.get_fe(),
    boundary_force.get_face_quadrature(),
    fdl::compute_flag_dependencies(face_me_flags));
  fdl::MechanicsValues<dim, spacedim> m_values(fe_face_values,
                                               position,
                                               velocity,
                                               face_me_flags);

  std::vector<Tensor<1, spacedim>> face_forces(
    boundary_force.get_face_quadrature().size());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            fe_face_values.reinit(cell, face);
            m_values.reinit(cell);
            auto view = make_array_view(face_forces);
            boundary_force.compute_boundary_force(time, m_values, face, view);
            output << "face = " << face << " forces = ";
            for (const Tensor<1, spacedim> &f : face_forces)
              output << "  " << f << '\n';
          }
    }
}

template <int dim>
class Rotate45 : public Function<dim>
{
public:
  Rotate45()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, n_components());

    const Tensor<2, dim> matrix =
      Physics::Transformations::Rotations::rotation_matrix_2d(numbers::PI /
                                                              4.0);
    return (matrix * p)[component];
  }
};

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "orthogonal_spring_dashpot_01.log");
  auto input_db = app_initializer->getInputDatabase();

  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
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

  LinearAlgebra::distributed::Vector<double> position(
    dof_handler.locally_owned_dofs(), comm);
  VectorTools::interpolate(dof_handler, Rotate45<2>(), position);
  LinearAlgebra::distributed::Vector<double> velocity(
    dof_handler.locally_owned_dofs(), comm);
  for (std::size_t i = 0; i < position.locally_owned_size(); ++i)
    velocity[i] = 1.0;

  // We have to do some manual setup of stuff normally done inside the library
  MappingCartesian<2> mapping;
  QMidpoint<1>        face_quadrature;

  const double spring_constant  = 10.0;
  const double damping_constant = 5.0;

  std::ofstream output("output");

  {
    fdl::OrthogonalSpringDashpotForce<2> boundary_force(face_quadrature,
                                                        spring_constant,
                                                        damping_constant,
                                                        dof_handler,
                                                        mapping,
                                                        Rotate45<2>());
    boundary_force.set_reference_position(reference);
    output << "test 1:\ndisplace (2, 2), spring constant = " << spring_constant
           << "\n"
           << "velocity (1, 1), damping constant = " << damping_constant << '\n'
           << "uses a position function\n";
    print_force(boundary_force, dof_handler, 0.0, position, velocity, output);
  }

  output << "\n\n";

  {
    fdl::OrthogonalSpringDashpotForce<2> boundary_force(face_quadrature,
                                                        spring_constant,
                                                        damping_constant,
                                                        dof_handler,
                                                        position);
    boundary_force.set_reference_position(reference);
    output << "test 2:\ndisplace (2, 2), spring constant = " << spring_constant
           << "\n"
           << "velocity (1, 1), damping constant = " << damping_constant << '\n'
           << "uses a FE position vector\n";
    print_force(boundary_force, dof_handler, 0.0, position, velocity, output);
  }

  output << "\n\n";

  {
    fdl::OrthogonalSpringDashpotForce<2> boundary_force(face_quadrature,
                                                        spring_constant,
                                                        damping_constant,
                                                        dof_handler,
                                                        position,
                                                        {0u, 3u});
    boundary_force.set_reference_position(reference);
    output << "test 3:\ndisplace (2, 2), spring constant = " << spring_constant
           << "\n"
           << "velocity (1, 1), damping constant = " << damping_constant << '\n'
           << "uses a FE position vector\n"
           << "only on left and top boundaries\n";
    print_force(boundary_force, dof_handler, 0.0, position, velocity, output);
  }

  output << "\n\n";

  {
    fdl::OrthogonalSpringDashpotForce<2> boundary_force(face_quadrature,
                                                        spring_constant,
                                                        damping_constant,
                                                        dof_handler,
                                                        position,
                                                        {numbers::invalid_boundary_id});
    boundary_force.set_reference_position(reference);
    output << "test 4:\ndisplace (2, 2), spring constant = " << spring_constant
           << "\n"
           << "velocity (1, 1), damping constant = " << damping_constant << '\n'
           << "uses a FE position vector\n"
           << "on no boundaries\n";
    print_force(boundary_force, dof_handler, 0.0, position, velocity, output);
  }

  output << "\n\n";

  {
    const double load_time    = 1.0;
    const double loaded_force = 10.0;

    fdl::OrthogonalLinearLoadForce<2> boundary_force(face_quadrature,
                                                     load_time,
                                                     loaded_force);
    output << "test 5:\ndisplace (2, 2), load_time = " << load_time << "\n"
           << "loaded force = " << loaded_force << '\n';
    print_force(boundary_force, dof_handler, 0.0, position, velocity, output);
  }

  output << "\n\n";

  {
    const double load_time    = 1.0;
    const double loaded_force = 10.0;

    fdl::OrthogonalLinearLoadForce<2> boundary_force(face_quadrature,
                                                     load_time,
                                                     loaded_force);
    output << "test 6:\ndisplace (2, 2), load_time = " << load_time << "\n"
           << "loaded force = " << loaded_force << '\n';
    print_force(boundary_force, dof_handler, 0.5, position, velocity, output);
  }

  output << "\n\n";

  {
    const double load_time    = 1.0;
    const double loaded_force = 10.0;

    fdl::OrthogonalLinearLoadForce<2> boundary_force(face_quadrature,
                                                     load_time,
                                                     loaded_force,
                                                     {0u, 1u});
    output << "test 7:\ndisplace (2, 2), load_time = " << load_time << "\n"
           << "loaded force = " << loaded_force << '\n'
           << "only on left and right boundaries\n";
    print_force(boundary_force, dof_handler, 1.0, position, velocity, output);
  }
}
