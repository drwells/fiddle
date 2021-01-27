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

using namespace dealii;

template <int dim>
class Identity : public Function<dim>
{
public:
  Identity() : Function<dim>(dim)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    AssertIndexRange(component, dim);
    return p[component];
  }
};

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
  parallel::shared::Triangulation<2> native_tria(mpi_comm);
  GridGenerator::hyper_ball(native_tria);
  native_tria.refine_global(1);
  for (unsigned int i = 0; i < 3; ++i)
    {
      for (const auto &cell : native_tria.active_cell_iterators())
        if (cell->barycenter()[0] > 0.0)
          cell->set_refine_flag();
      native_tria.execute_coarsening_and_refinement();
    }

  {
    GridOut       go;
    std::ofstream out("tria-1.eps");
    go.write_eps(native_tria, out);
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
  auto native_position_partitioner = std::make_shared<Utilities::MPI::Partitioner>
    (native_position_dh.locally_owned_dofs(),
     locally_relevant_position_dofs,
     mpi_comm);
  LinearAlgebra::distributed::Vector<double> native_initial_position
    (native_position_partitioner);
  LinearAlgebra::distributed::Vector<double> native_current_position
    (native_position_partitioner);
  VectorTools::interpolate(native_position_dh, Identity<2>(), native_initial_position);
  VectorTools::interpolate(native_position_dh, Displace<2>(), native_current_position);
  native_initial_position.update_ghost_values();
  native_current_position.update_ghost_values();
  // TODO: get the rest of this pipeline working with the displaced field.
  // Ultimately we want to be able to plot visit data with those coordinates and
  // not the actual triangulation coordinates.

  MappingFEField<2, 2, decltype(native_current_position)> native_mapping(
    native_position_dh, native_current_position);

  fdl::FEIntersectionPredicate<2> fe_pred({bbox},
                                          mpi_comm,
                                          native_position_dh,
                                          native_mapping);

  {
    BoundingBoxDataOut<2> bbox_data_out;
    bbox_data_out.build_patches(fe_pred.active_cell_bboxes);
    std::ofstream out("bboxes-" + std::to_string(rank) + ".vtk");
    bbox_data_out.write_vtk(out);
  }

  // set up the overlap tria:
  fdl::OverlapTriangulation<2> overlap_tria(native_tria, fe_pred);
  DoFHandler<2> overlap_position_dh(overlap_tria);
  overlap_position_dh.distribute_dofs(position_fe);

  Vector<double> overlap_current_position(overlap_position_dh.n_dofs());

  const std::vector<types::global_dof_index> overlap_to_native_position =
    fdl::compute_overlap_to_native_dof_translation(overlap_tria,
                                                   overlap_position_dh,
                                                   native_position_dh);
  fdl::Scatter<double> position_scatter(overlap_to_native_position,
                                        native_position_dh.locally_owned_dofs(),
                                        mpi_comm);
  position_scatter.global_to_overlap_start(native_current_position, 0,
                                           overlap_current_position);
  position_scatter.global_to_overlap_finish(native_current_position,
                                            overlap_current_position);
  MappingFEField<2, 2, decltype(overlap_current_position)> overlap_mapping(
    overlap_position_dh, overlap_current_position);

  {
    GridOut       go;
    std::ofstream out("overlap-tria-" + std::to_string(rank) + ".eps");
    go.write_eps(overlap_tria, out);
  }

  // Now the topology and geometry are ready. set up dofs:
  FE_Q<2>       fe(3);
  DoFHandler<2> native_dof_handler(native_tria);
  native_dof_handler.distribute_dofs(fe);
  DoFHandler<2> overlap_dof_handler(overlap_tria);
  overlap_dof_handler.distribute_dofs(fe);

  // TODO: VT::interpolate requires ghost data with LA::d::V, not sure why
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(native_dof_handler,
                                          locally_relevant_dofs);
  LinearAlgebra::distributed::Vector<double> native_solution(
    native_dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
  VectorTools::interpolate(native_dof_handler,
                           Functions::CosineFunction<2>(),
                           native_solution);

  Vector<double> overlap_solution(overlap_dof_handler.n_dofs());

  const std::vector<types::global_dof_index> overlap_to_native_solution =
    fdl::compute_overlap_to_native_dof_translation(overlap_tria,
                                                   overlap_dof_handler,
                                                   native_dof_handler);
  fdl::Scatter<double> scatter(overlap_to_native_solution,
                               native_dof_handler.locally_owned_dofs(),
                               mpi_comm);
  // Scatter forward...
  scatter.global_to_overlap_start(native_solution, 0, overlap_solution);
  scatter.global_to_overlap_finish(native_solution, overlap_solution);

  // and back.
  scatter.overlap_to_global_start(overlap_solution,
                                  VectorOperation::insert,
                                  0,
                                  native_solution);
  scatter.overlap_to_global_finish(overlap_solution,
                                   VectorOperation::insert,
                                   native_solution);

  // output native data:
  {
    LinearAlgebra::distributed::Vector<double> ghosted_native_solution(
      native_dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
    ghosted_native_solution = native_solution;
    ghosted_native_solution.update_ghost_values();

    DataOut<2> data_out;
    data_out.attach_dof_handler(native_dof_handler);
    data_out.add_data_vector(ghosted_native_solution, "solution");
    data_out.build_patches(native_mapping);

    data_out.write_vtu_with_pvtu_record("./", "solution", 0, mpi_comm);
  }

  // output overlap data:
  {
    DataOut<2> data_out;
    data_out.attach_dof_handler(overlap_dof_handler);
    data_out.add_data_vector(overlap_solution, "solution");
    data_out.build_patches(overlap_mapping);

    std::ofstream out("overlap-solution-" + std::to_string(rank) + ".vtu");
    data_out.write_vtu(out);
  }
}
