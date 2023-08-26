#include <fiddle/grid/grid_utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "../tests.h"

// Verify that the centroid utility function works

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
  GridGenerator::hyper_cube(tria, 0, 1, true);
  tria.refine_global(4);

  const auto centroid_0 = fdl::compute_centroid(MappingQ1<2>(), tria, {0u});
  const auto centroid_1 = fdl::compute_centroid(MappingQ1<2>(), tria, {1u});
  const auto centroid_01 =
    fdl::compute_centroid(MappingQ1<2>(), tria, {0u, 1u});
  // make sure things work with unsorted, weird input too
  const auto centroid_all =
    fdl::compute_centroid(MappingQ1<2>(), tria, {4u, 3u, 2u, 0u, 4u, 1u});

  std::ofstream output;
  if (rank == 0)
    {
      output.open("output");
      output << "rank = " << rank << '\n';
      output << "centroids: " << '\n'
             << centroid_0 << '\n'
             << centroid_1 << '\n'
             << centroid_01 << '\n'
             << centroid_all << '\n';
    }
}
