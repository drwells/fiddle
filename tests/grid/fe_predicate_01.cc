#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/intersection_predicate.h>
#include <fiddle/grid/overlap_tria.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

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

#include <cmath>
#include <fstream>
#include <iostream>

// Test the bounding boxes per element formerly computed by fe_predicate and now
// by compute_cell_bboxes

using namespace dealii;

template <int dim>
class Displace : public Function<dim>
{
public:
  Displace()
    : Function<dim>(dim)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    AssertIndexRange(component, dim);
    return p[component] + p[0];
  }
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const auto                       mpi_comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(mpi_comm);
  const auto partitioner =
    parallel::shared::Triangulation<2>::Settings::partition_zorder;
  parallel::shared::Triangulation<2> native_tria(mpi_comm,
                                                 {},
                                                 false,
                                                 partitioner);
  GridGenerator::hyper_ball(native_tria);
  for (unsigned int i = 0; i < 3; ++i)
    {
      for (const auto &cell : native_tria.active_cell_iterators())
        if (cell->barycenter()[0] > 0.0)
          cell->set_refine_flag();
      native_tria.execute_coarsening_and_refinement();
    }

  BoundingBox<2> bbox;
  switch (rank)
    {
      case 0:
        bbox = BoundingBox<2>(
          std::make_pair(Point<2>(0.0, 0.0), Point<2>(2.0, 2.0)));
        break;
      case 1:
        bbox = BoundingBox<2>(
          std::make_pair(Point<2>(-2.0, 0.0), Point<2>(0.0, 2.0)));
        break;
      case 2:
        bbox = BoundingBox<2>(
          std::make_pair(Point<2>(-2.0, -2.0), Point<2>(0.0, 0.0)));
        break;
      case 3:
        bbox = BoundingBox<2>(
          std::make_pair(Point<2>(0.0, -2.0), Point<2>(2.0, 0.0)));
        break;
      case 4:
        bbox = BoundingBox<2>(
          std::make_pair(Point<2>(-0.5, -0.5), Point<2>(0.5, 0.5)));
        break;
      default:
        break;
        // Assert(false, ExcNotImplemented());
    }

  // set up the position system, which describes the geometry:
  FESystem<2>   position_fe(FE_Q<2>(1), 2);
  DoFHandler<2> native_position_dh(native_tria);
  native_position_dh.distribute_dofs(position_fe);

  IndexSet locally_relevant_position_dofs;
  DoFTools::extract_locally_relevant_dofs(native_position_dh,
                                          locally_relevant_position_dofs);
  auto native_position_partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(
      native_position_dh.locally_owned_dofs(),
      locally_relevant_position_dofs,
      mpi_comm);
  LinearAlgebra::distributed::Vector<double> native_current_position(
    native_position_partitioner);
  VectorTools::interpolate(native_position_dh,
                           Displace<2>(),
                           native_current_position);
  native_current_position.update_ghost_values();

  MappingFEField<2, 2, decltype(native_current_position)> native_mapping(
    native_position_dh, native_current_position);

  const auto bboxes =
    fdl::compute_cell_bboxes<2, 2, float>(native_position_dh, native_mapping);

  {
    BoundingBoxDataOut<2> bbox_data_out;
    bbox_data_out.build_patches(bboxes);
    // TODO: generalize so we can run this in parallel
    std::ofstream         out("output");
    DataOutBase::VtkFlags flags;
    flags.print_date_and_time = false;
    bbox_data_out.set_flags(flags);
    bbox_data_out.write_vtk(out);
  }
}
