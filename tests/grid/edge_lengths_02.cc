#include <fiddle/grid/grid_utilities.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "../tests.h"

// Verify the output of the edge length computation in 3D.

int
main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const int  rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto partitioner =
    parallel::shared::Triangulation<3>::Settings::partition_zorder;
  parallel::shared::Triangulation<3> tria(MPI_COMM_WORLD,
                                          {},
                                          true,
                                          partitioner);
  GridGenerator::hyper_ball(tria);
  tria.refine_global(1);

  const std::vector<float> edge_lengths =
    fdl::compute_longest_edge_lengths(tria, MappingQ1<3>(), QMidpoint<1>());

  unsigned int local_index = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          float measure = 0.0;
          for (const auto &line_index : cell->line_indices())
            measure =
              std::max<float>(measure, cell->line(line_index)->measure());

          AssertIndexRange(local_index, edge_lengths.size());
          AssertThrow(std::abs(measure - edge_lengths[local_index]) <
                        measure * 1e-7,
                      ExcMessage("measure should be close"));
          ++local_index;
        }
    }

#if 0
  Vector<float> edge_lengths_2(edge_lengths.begin(), edge_lengths.end());
  DataOut<3>    data_out;
  data_out.attach_triangulation(tria);
  data_out.add_data_vector(edge_lengths_2, "L", DataOut<3>::type_cell_data);

  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record("./", "solution", 0, MPI_COMM_WORLD, 8);
#endif

  std::ostringstream out;
  out << "rank = " << rank << '\n';
  for (const auto &f : edge_lengths)
    out << f << '\n';

  std::ofstream output;
  if (rank == 0)
    output.open("output");

  print_strings_on_0(out.str(), MPI_COMM_WORLD, output);
}
