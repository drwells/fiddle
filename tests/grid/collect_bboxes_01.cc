#include <fiddle/grid/box_utilities.h>

#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/grid_generator.h>

#include <fstream>

#include "../tests.h"

template <int spacedim, typename Number1, typename Number2>
dealii::BoundingBox<spacedim, Number1>
convert(const dealii::BoundingBox<spacedim, Number2> &input)
{
  // We should get a better conversion constructor
  dealii::Point<spacedim, Number1> p0;
  dealii::Point<spacedim, Number1> p1;
  for (unsigned int d = 0; d < spacedim; ++d)
  {
    p0[d] = input.get_boundary_points().first[d];
    p1[d] = input.get_boundary_points().second[d];
  }

  return dealii::BoundingBox<spacedim, Number1>(std::make_pair(p0, p1));
}

// Test collect_bboxes in parallel
template <int spacedim, typename Number>
void test()
{
  using namespace dealii;

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);
  const auto n_procs  = Utilities::MPI::n_mpi_processes(mpi_comm);

  parallel::shared::Triangulation<spacedim> tria(mpi_comm);
  GridGenerator::hyper_ball(tria);
  tria.refine_global(1);

  std::vector<BoundingBox<spacedim, Number>> bboxes;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      bboxes.emplace_back(convert<spacedim, Number, double>(cell->bounding_box()));

  auto all_bboxes = fdl::collect_all_active_cell_bboxes(tria, bboxes);

  for (const auto &cell : tria.active_cell_iterators())
  {
    // we can skip the locality check now
    AssertThrow((convert<spacedim, Number, double>(cell->bounding_box()) ==
                 all_bboxes[cell->active_cell_index()]),
                ExcMessage("should be equal"));
  }

  std::ostringstream this_proc_out;
  this_proc_out << "Rank = " << rank << '\n';

  DataOutBase::VtkFlags flags;
  flags.print_date_and_time = false;
  {
      this_proc_out << "Local bounding boxes:\n";
      BoundingBoxDataOut<spacedim> bbox_data_out;
      bbox_data_out.set_flags(flags);
      bbox_data_out.build_patches(bboxes);
      bbox_data_out.write_vtk(this_proc_out);
  }

  this_proc_out << "\n\n\n";

  {
      this_proc_out << "All bounding boxes:\n";
      BoundingBoxDataOut<spacedim> bbox_data_out;
      bbox_data_out.set_flags(flags);
      bbox_data_out.build_patches(all_bboxes);
      bbox_data_out.write_vtk(this_proc_out);
  }

  if (rank != n_procs - 1)
      this_proc_out << '\n';

  std::ofstream output;
  if (rank == 0)
      output.open("output");

  print_strings_on_0(this_proc_out.str(), mpi_comm, output);
}

int
main(int argc, char **argv)
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<NDIM, float>();
  test<NDIM, double>();
}
