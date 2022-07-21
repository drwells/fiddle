#include <fiddle/base/exceptions.h>

#include <fiddle/mechanics/mechanics_values.h>
#include <fiddle/mechanics/part.h>

#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>

#include <fstream>

#include "../tests.h"

// Test basic mechanics values stuff. Happens to use Part too.

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  // setup deal.II stuff:
  Triangulation<dim, spacedim> native_tria;
  GridGenerator::hyper_cube(native_tria);
  FESystem<dim, spacedim> fe(FE_Q<dim, spacedim>(2), spacedim);

  FunctionParser<spacedim> initial_position(
    extract_fp_string(input_db->getDatabase("test")->getDatabase("position")),
    "PI=" + std::to_string(numbers::PI),
    "X_0,X_1");

  // Now set up fiddle things for the test:
  fdl::Part<dim, spacedim> part(native_tria, fe, {}, initial_position);

  // and the test itself:
  {
    const auto &dof_handler = part.get_dof_handler();

    MappingQGeneric<dim, spacedim> mapping(1);
    QGauss<dim>                    quadrature(fe.degree + 1);

    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      quadrature,
                                      update_values | update_gradients);

    fdl::MechanicsValues<dim, spacedim> mechanics_values(fe_values,
                                                         part.get_position(),
                                                         part.get_velocity(),
                                                         fdl::update_FF |
                                                           fdl::update_det_FF);

    std::ofstream out("output");
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        mechanics_values.reinit(cell);

        out << "J:\n";
        for (const double &J : mechanics_values.get_det_FF())
          out << J << '\n';

        out << "FF:\n";
        for (const Tensor<2, spacedim> &FF : mechanics_values.get_FF())
          out << FF << '\n';
      }

    fdl::MechanicsUpdateFlags face_me_flags =
      fdl::update_deformed_normal_vectors;

    QGauss<dim - 1> face_quadrature(fe.degree + 1);
    FEFaceValues<dim, spacedim> fe_face_values(
      mapping, fe, face_quadrature, fdl::compute_flag_dependencies(face_me_flags));

    fdl::MechanicsValues<dim, spacedim> face_mechanics_values(fe_face_values,
                                                              part.get_position(),
                                                              part.get_velocity(),
                                                              face_me_flags);

    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            fe_face_values.reinit(cell, face);
            face_mechanics_values.reinit(cell);

            out << "deformed normal vectors:\n";
            for (const auto &n : face_mechanics_values.get_deformed_normal_vectors())
              out << n << '\n';
          }
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<2>(app_initializer);
}
