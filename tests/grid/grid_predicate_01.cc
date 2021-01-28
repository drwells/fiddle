#include <deal.II/base/bounding_box_data_out.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fiddle/grid/intersection_predicate.h>
#include <fiddle/grid/overlap_tria.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <cmath>
#include <fstream>
#include <iostream>

// Test the bounding boxes per element computed by TriaIntersectionPredicate

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const auto                       mpi_comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
  const auto n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);
  parallel::shared::Triangulation<2> native_tria(mpi_comm);
  GridGenerator::hyper_ball(native_tria);
  for (unsigned int i = 0; i < 4; ++i)
    {
      for (const auto &cell : native_tria.active_cell_iterators())
        if (cell->barycenter()[0] > 0.0)
          cell->set_refine_flag();
      native_tria.execute_coarsening_and_refinement();
    }

  std::vector<BoundingBox<2>> bboxes;
  switch (rank)
    {
      case 0:
        bboxes.emplace_back(std::make_pair(Point<2>(0.0, 0.0), Point<2>(2.0, 2.0)));
        DEAL_II_FALLTHROUGH;
      case 1:
        bboxes.emplace_back(
          std::make_pair(Point<2>(-2.0, 0.0), Point<2>(0.0, 2.0)));
        DEAL_II_FALLTHROUGH;
      case 2:
        bboxes.emplace_back(
          std::make_pair(Point<2>(-2.0, -2.0), Point<2>(0.0, 0.0)));
        break;
      case 3:
        bboxes.emplace_back(
          std::make_pair(Point<2>(0.0, -2.0), Point<2>(2.0, 0.0)));
        break;
      case 4:
        bboxes.emplace_back(
          std::make_pair(Point<2>(-0.5, -0.5), Point<2>(0.5, 0.5)));
        break;
      default:
        Assert(false, ExcNotImplemented());
    }

  fdl::TriaIntersectionPredicate<2> tria_pred(bboxes);
  fdl::OverlapTriangulation<2> overlap_tria(native_tria, tria_pred);

  {
    std::ofstream out("output-" + std::to_string(rank));
    GridOut       go;
    go.write_vtk(overlap_tria, out);
  }

  MPI_Barrier(mpi_comm);

  if (rank == 0)
  {
    std::ofstream out("output");
    for (unsigned int r = 0; r < n_procs; ++r)
    {
      std::ifstream in("output-" + std::to_string(r));
      out << in.rdbuf() << "\n";
    }
  }
}
