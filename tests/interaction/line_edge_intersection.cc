#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

using namespace dealii;
using namespace SAMRAI;

// Test the line edge intersection code

void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto                input_db = app_initializer->getInputDatabase();
  auto                test_db  = input_db->getDatabase("test");
  Triangulation<2>    tria_bulk;
  Triangulation<1, 2> tria;
  GridGenerator::hyper_ball(tria_bulk);
  GridGenerator::hyper_sphere(
    tria,
    Point<2>(test_db->getDoubleWithDefault("circle_center_x_coordinate", 0.0),
             test_db->getDoubleWithDefault("circle_center_y_coordinate", 0.0)),
    test_db->getDoubleWithDefault("circle_radius", 0.0));
  tria.refine_global(6);
  const Point<2> r(
    test_db->getDoubleWithDefault("stencil_center_x_coordinate", 0.0),
    test_db->getDoubleWithDefault("stencil_center_y_coordinate", 0.0));
  const MappingQ<1, 2> mapping(1);
  std::vector<double>  t_vals;
  FE_Nothing<1, 2>     fe;
  DoFHandler<1, 2>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  std::ofstream output("output");
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::array<Point<2>, 2> Pts = {
        mapping.transform_unit_to_real_cell(cell, Point<1>(0)),
        mapping.transform_unit_to_real_cell(cell, Point<1>(1))};
      std_cxx17::optional<double> convex_coef =
        fdl::intersect_stencil_with_simplex<1>(
          Pts,
          r,
          test_db->getDoubleWithDefault("stencil_width", 0.0),
          test_db->getIntegerWithDefault("stencil_axis", 0));
      if (convex_coef)
        {
          output << "The intersection convex coefficient is: " << *convex_coef
                 << '\n';
        }
    }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "line_edge_intersection.log");
  test(app_initializer);
}
