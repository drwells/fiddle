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

// Test part serialization

using namespace dealii;
using namespace SAMRAI;

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  // setup deal.II stuff:
  const auto partitioner =
    parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder;
  parallel::shared::Triangulation<dim, spacedim> native_tria(MPI_COMM_WORLD,
                                                             {},
                                                             false,
                                                             partitioner);
  GridGenerator::hyper_cube(native_tria);
  native_tria.refine_global(3);
  FESystem<dim, spacedim> fe(FE_Q<dim, spacedim>(2), spacedim);

  FunctionParser<spacedim> initial_position(
    extract_fp_string(input_db->getDatabase("test")->getDatabase("position")),
    "PI=" + std::to_string(numbers::PI),
    "X_0,X_1");

  FunctionParser<spacedim> initial_velocity(
    extract_fp_string(input_db->getDatabase("test")->getDatabase("velocity")),
    "PI=" + std::to_string(numbers::PI),
    "X_0,X_1");

  // set up fiddle stuff for the test:
  fdl::Part<dim, spacedim> part_0(
    native_tria, fe, {}, initial_position, initial_velocity);
  fdl::Part<dim, spacedim> part_1(native_tria, fe);

  // and the test itself:
  std::string serialization;
  {
    std::ostringstream              out_str;
    boost::archive::binary_oarchive oarchive(out_str);
    part_0.save(oarchive, 0);
    serialization = out_str.str();
  }

  {
    std::istringstream              in_str(serialization);
    boost::archive::binary_iarchive iarchive(in_str);
    part_1.load(iarchive, 0);
  }

  auto temp = part_0.get_position();
  temp -= part_1.get_position();
  auto temp1 = part_0.get_velocity();
  temp1 -= part_1.get_velocity();

  const double l2_1      = part_0.get_position().l2_norm();
  const double l2_2      = part_0.get_velocity().l2_norm();
  const double l2_diff_1 = temp.l2_norm();
  const double l2_diff_2 = temp1.l2_norm();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream output("output");
      // TODO - this might not be consistent between different boost versions,
      // but it should be since we only serialize double arrays.
      output << "string size = " << serialization.size() << std::endl;
      output << "norm = " << l2_1 << std::endl;
      output << "difference norm = " << l2_diff_1 << std::endl;
      output << "norm = " << l2_2 << std::endl;
      output << "difference norm = " << l2_diff_2 << std::endl;
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
