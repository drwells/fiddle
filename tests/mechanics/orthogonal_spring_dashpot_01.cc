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

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>
#include <memory>
#include <vector>

// Basic test for OrthogonalSpringDashpotForce

using namespace dealii;
template <int dim>
class IP2 : public Function<dim>
{
public:
  IP2()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);
    return p[component] + 2.0;
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

  LinearAlgebra::distributed::Vector<double> current_position(
    dof_handler.locally_owned_dofs(), comm);
  LinearAlgebra::distributed::Vector<double> current_velocity(
    dof_handler.locally_owned_dofs(), comm);
  for (std::size_t i = 0; i < current_position.locally_owned_size(); ++i)
    {
      current_position[i] = 2.0 + reference[i];
      current_velocity[i] = 1.0;
    }

  // We have to do some manual setup of stuff normally done inside the library
  fdl::MechanicsUpdateFlags face_me_flags =
    fdl::MechanicsUpdateFlags::update_deformed_normal_vectors |
    fdl::MechanicsUpdateFlags::update_velocity_values;
  MappingCartesian<2>     mapping;
  QMidpoint<1>            face_quadrature;
  FEFaceValues<2>         fe_face_values(mapping,
                                 fe,
                                 face_quadrature,
                                 fdl::compute_flag_dependencies(face_me_flags));
  fdl::MechanicsValues<2> m_values(fe_face_values,
                                   current_position,
                                   current_velocity,
                                   face_me_flags);

  const double spring_constant  = 10.0;
  const double damping_constant = 5.0;

  std::unique_ptr<fdl::OrthogonalSpringDashpotForce<2>>
    orthogonal_spring_dashpot_force;

  if (input_db->getBoolWithDefault("use_function", false))
    orthogonal_spring_dashpot_force =
      std::make_unique<fdl::OrthogonalSpringDashpotForce<2>>(face_quadrature,
                                                             spring_constant,
                                                             damping_constant,
                                                             dof_handler,
                                                             mapping,
                                                             IP2<2>());
  else
    orthogonal_spring_dashpot_force =
      std::make_unique<fdl::OrthogonalSpringDashpotForce<2>>(face_quadrature,
                                                             spring_constant,
                                                             damping_constant,
                                                             dof_handler,
                                                             current_position);

  orthogonal_spring_dashpot_force->set_reference_position(reference);

  orthogonal_spring_dashpot_force->setup_force(0.0,
                                               current_position,
                                               current_velocity);
  std::vector<Tensor<1, 2>> face_forces(face_quadrature.size());
  std::ofstream             output("output");
  output << "test 1:\ndisplace (2, 2), spring constant = " << spring_constant
         << "\n"
         << "velocity (1, 1), damping constant = " << damping_constant << '\n';
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            fe_face_values.reinit(cell, face);
            m_values.reinit(cell);
            auto view = make_array_view(face_forces);
            orthogonal_spring_dashpot_force->compute_boundary_force(0.0,
                                                                    m_values,
                                                                    face,
                                                                    view);
            output << "forces = ";
            for (const Tensor<1, 2> &f : face_forces)
              output << "  " << f << '\n';
          }
    }
}
