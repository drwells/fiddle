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

// Test the line face intersection code

void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto                input_db = app_initializer->getInputDatabase();
  auto                test_db  = input_db->getDatabase("test");
  Triangulation<2, 3> tria;
  GridGenerator::hyper_sphere(
    tria,
    Point<3>(test_db->getDoubleWithDefault("circle_center_x_coordinate", 0.0),
             test_db->getDoubleWithDefault("circle_center_y_coordinate", 0.0),
             test_db->getDoubleWithDefault("circle_center_z_coordinate", 0.0)),
    test_db->getDoubleWithDefault("sphere_radius", 0.0));
  tria.refine_global(5);
  dealii::Point<3> r(
    test_db->getDoubleWithDefault("stencil_center_x_coordinate", 0.0),
    test_db->getDoubleWithDefault("stencil_center_y_coordinate", 0.0),
    test_db->getDoubleWithDefault("stencil_center_z_coordinate", 0.0));
  const MappingQ<2, 3> mapping(1);
  FE_Nothing<2, 3>     fe;
  DoFHandler<2, 3>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  std::ofstream output("output");
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::array<Point<3>, 3> Pts = {
        mapping.transform_unit_to_real_cell(cell, Point<2>(0, 0)),
        mapping.transform_unit_to_real_cell(cell, Point<2>(0, 1)),
        mapping.transform_unit_to_real_cell(cell, Point<2>(1, 0))};
      std_cxx17::optional<double> convex_coef =
        fdl::intersect_stencil_with_simplex<2>(
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
    new IBTK::AppInitializer(argc, argv, "line_face_intersection.log");
  test(app_initializer);
}
