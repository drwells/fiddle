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

// More MechanicsValues output

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

    const auto flags =
      fdl::update_right_cauchy_green | fdl::update_first_invariant |
      fdl::update_second_invariant | fdl::update_third_invariant;
    fdl::MechanicsValues<dim, spacedim> mechanics_values(fe_values,
                                                         part.get_position(),
                                                         part.get_velocity(),
                                                         flags);

    std::ofstream out("output");
    out << "Basic quantities\n";
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

        out << "C:\n";
        for (const SymmetricTensor<2, spacedim> &C :
             mechanics_values.get_right_cauchy_green())
          out << C << '\n';

        out << "I_1:\n";
        for (const double &I_1 : mechanics_values.get_first_invariant())
          out << I_1 << '\n';

        out << "I_2:\n";
        for (const double &I_2 : mechanics_values.get_second_invariant())
          out << I_2 << '\n';

        out << "I_3:\n";
        for (const double &I_3 : mechanics_values.get_third_invariant())
          out << I_3 << '\n';
      }

    out << "\nmore advanced quantities:\n";
    const std::vector<fdl::MechanicsUpdateFlags> me_flags{
      fdl::update_log_det_FF,
      fdl::update_green,
      fdl::update_modified_first_invariant,
      fdl::update_modified_second_invariant,
      fdl::update_first_invariant_dFF,
      fdl::update_modified_first_invariant_dFF};
    for (const auto me_flag : me_flags)
      {
        FEValues<dim, spacedim> fe_values(
          mapping, fe, quadrature, fdl::compute_flag_dependencies(me_flag));

        fdl::MechanicsValues<dim, spacedim> mechanics_values(
          fe_values, part.get_position(), part.get_velocity(), me_flag);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            fe_values.reinit(cell);
            mechanics_values.reinit(cell);

            switch (me_flag)
              {
                case fdl::update_log_det_FF:
                  out << "log(det(FF)):\n";
                  for (const double &l :
                       mechanics_values.get_log_det_FF())
                    out << l << '\n';
                  break;
                case fdl::update_green:
                  out << "E:\n";
                  for (const SymmetricTensor<spacedim, 2> &E :
                       mechanics_values.get_green())
                    out << E << '\n';
                  break;
                case fdl::update_modified_first_invariant:
                  out << "I1_bar:\n";
                  for (const double &I1_bar :
                       mechanics_values.get_modified_first_invariant())
                    out << I1_bar << '\n';
                  break;
                case fdl::update_modified_second_invariant:
                  out << "I2_bar:\n";
                  for (const double &I2_bar :
                       mechanics_values.get_modified_second_invariant())
                    out << I2_bar << '\n';
                  break;
                case fdl::update_first_invariant_dFF:
                  out << "I1_dFF:\n";
                  for (const Tensor<spacedim, 2> &t :
                       mechanics_values.get_first_invariant_dFF())
                    out << t << '\n';
                  break;
                case fdl::update_modified_first_invariant_dFF:
                  out << "I1_bar_dFF:\n";
                  for (const Tensor<spacedim, 2> &m :
                       mechanics_values.get_modified_first_invariant_dFF())
                    out << m << '\n';
                  break;
                default:
                  AssertThrow(false, fdl::ExcFDLInternalError());
              }
          }
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
