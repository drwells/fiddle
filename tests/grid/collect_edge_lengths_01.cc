#include <fiddle/grid/grid_utilities.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "../tests.h"

// Verify the output of the collected edge length computation. Should match the
// serial edge_lengths_01 test.

int
main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const int  rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto partitioner =
    parallel::shared::Triangulation<2>::Settings::partition_zorder;
  parallel::shared::Triangulation<2> tria(MPI_COMM_WORLD,
                                          {},
                                          false,
                                          partitioner);
  GridGenerator::hyper_ball(tria);
  tria.refine_global(3);

  const std::vector<float> edge_lengths =
    fdl::compute_longest_edge_lengths(tria, MappingQ1<2>(), QGauss<1>(3));

  const std::vector<float> all_edge_lengths =
    fdl::collect_longest_edge_lengths(tria, edge_lengths);

  std::ofstream output;
  if (rank == 0)
    {
      output.open("output");
      output << "rank = " << rank << '\n';
      for (const auto &f : all_edge_lengths)
        output << f << '\n';
    }
}
