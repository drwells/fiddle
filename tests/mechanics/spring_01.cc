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

// Basic test for SpringForce

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


using namespace dealii;
template <int dim>
class Shift : public Function<dim>
{
public:
  Shift()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    AssertIndexRange(component, dim);
    return p[component] + p[0];
  }
};


int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "spring_01.log");
  auto input_db = app_initializer->getInputDatabase();

  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  if (input_db->getBoolWithDefault("multiple_materials", false))
    {
      auto cell = tria.begin_active();
      cell->set_material_id(42);
      ++cell;
      cell->set_material_id(99);
    }
  FESystem<2>   fe(FE_Q<2>(1), 2);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const MPI_Comm comm = MPI_COMM_WORLD;

  LinearAlgebra::distributed::Vector<double> reference(
    dof_handler.locally_owned_dofs(), comm);
  VectorTools::interpolate(dof_handler,
                           Functions::IdentityFunction<2>(),
                           reference);

  LinearAlgebra::distributed::Vector<double> current(
    dof_handler.locally_owned_dofs(), comm);
  for (std::size_t i = 0; i < current.locally_owned_size(); ++i)
    current[i] = 2.0 + reference[i];

  // We have to do some manual setup of stuff normally done inside the library
  MappingCartesian<2> mapping;
  QMidpoint<2>        quadrature;
  const double        spring_constant = 10.0;

  std::unique_ptr<fdl::SpringForce<2>> spring_force;
  if (input_db->getBoolWithDefault("multiple_materials", false))
    {
      std::vector<types::material_id> materials;
      if (input_db->getBoolWithDefault("use_no_materials", false))
        materials = {numbers::invalid_material_id};
      else
        materials = {42, 99, 99, 99, 42};
      if (input_db->getBoolWithDefault("use_function", false))
        {
          spring_force = std::make_unique<fdl::SpringForce<2>>(quadrature,
                                                               spring_constant,
                                                               dof_handler,
                                                               mapping,
                                                               IP2<2>(),
                                                               materials);
          spring_force->set_reference_position(reference);
        }
      else
        // use the reference configuration in one of the tests
        spring_force = std::make_unique<fdl::SpringForce<2>>(quadrature,
                                                             spring_constant,
                                                             materials);
    }
  else
    {
      if (input_db->getBoolWithDefault("use_function", false))
        spring_force = std::make_unique<fdl::SpringForce<2>>(
          quadrature, spring_constant, dof_handler, mapping, IP2<2>());
      else
        spring_force = std::make_unique<fdl::SpringForce<2>>(quadrature,
                                                             spring_constant,
                                                             dof_handler,
                                                             current);
      spring_force->set_reference_position(reference);
    }

  FEValues<2>             fe_values(mapping,
                        fe,
                        quadrature,
                        spring_force->get_update_flags());
  fdl::MechanicsValues<2> m_values(fe_values,
                                   current,
                                   current,
                                   spring_force->get_mechanics_update_flags());

  spring_force->setup_force(0.0, current, current);
  std::vector<Tensor<1, 2>> forces(quadrature.size());
  std::ofstream             output("output");
  output << "test 1: displace 2 forward, spring constant = " << spring_constant
         << '\n';
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // more manual stuff that normally would not be necessary
      fe_values.reinit(cell);
      m_values.reinit(cell);
      output << "cell center = " << cell->center() << '\n';
      auto view = make_array_view(forces);
      spring_force->compute_volume_force(0.0, m_values, cell, view);
      output << "forces = ";
      for (const Tensor<1, 2> &f : forces)
        output << f << '\n';
    }

  VectorTools::interpolate(dof_handler, Shift<2>(), current);
  output << "test 2: displace + x forward, spring constant = "
         << spring_constant << '\n';
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // more manual stuff that normally would not be necessary
      fe_values.reinit(cell);
      m_values.reinit(cell);
      output << "cell center = " << cell->center() << '\n';
      auto view = make_array_view(forces);
      spring_force->compute_volume_force(0.0, m_values, cell, view);
      output << "forces = ";
      for (const Tensor<1, 2> &f : forces)
        output << f << '\n';
    }
}
