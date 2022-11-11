#include <fiddle/base/exceptions.h>

#include <fiddle/grid/overlap_tria.h>
#include <fiddle/grid/patch_map.h>

#include <fiddle/interaction/interaction_base.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>
#include <fiddle/transfer/scatter.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/SAMRAIDataCache.h>

#include <fstream>

#include "../tests.h"

using namespace dealii;
using namespace SAMRAI;

// First test driver for InteractionBase. Used to test
//
// 1. element coordinate scattering

template <int dim>
class Identity : public Function<dim>
{
public:
  Identity()
    : Function<dim>(dim)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    AssertIndexRange(component, dim);
    return p[component];
  }
};

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup deal.II stuff:
  parallel::shared::Triangulation<dim, spacedim> native_tria(mpi_comm);
  GridGenerator::concentric_hyper_shells(
    native_tria, Point<spacedim>(), 0.0625, 0.5, 2, 0.0);
  native_tria.refine_global(3);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);

  // Now set up fiddle things for the test:
  std::vector<BoundingBox<spacedim, float>> cell_bboxes;
  for (const auto &cell : native_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      cell_bboxes.push_back(fbbox);
    }

  const auto level_number = patch_hierarchy->getFinestLevelNumber();
  fdl::InteractionBase<dim, spacedim> interaction_base(
    input_db,
    native_tria,
    cell_bboxes,
    {}, // This class doesn't read edge lengths
    patch_hierarchy,
    std::make_pair(level_number, level_number));

  FESystem<dim>             position_fe(FE_Q<dim>(1), dim);
  DoFHandler<dim, spacedim> position_dof_handler(native_tria);
  position_dof_handler.distribute_dofs(position_fe);
  interaction_base.add_dof_handler(position_dof_handler);

  IndexSet locally_relevant_position_dofs;
  DoFTools::extract_locally_relevant_dofs(position_dof_handler,
                                          locally_relevant_position_dofs);
  auto position_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    position_dof_handler.locally_owned_dofs(),
    locally_relevant_position_dofs,
    mpi_comm);
  LinearAlgebra::distributed::Vector<double> position(position_partitioner);
  VectorTools::interpolate(position_dof_handler, Identity<dim>(), position);
  position.update_ghost_values();

  FESystem<dim>             F_fe(FE_DGQ<dim>(0), dim);
  DoFHandler<dim, spacedim> F_dof_handler(native_tria);
  F_dof_handler.distribute_dofs(F_fe);
  const MappingQ<dim> F_mapping(1);
  interaction_base.add_dof_handler(F_dof_handler);

  IndexSet locally_relevant_F_dofs;
  DoFTools::extract_locally_relevant_dofs(F_dof_handler,
                                          locally_relevant_F_dofs);
  auto F_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    F_dof_handler.locally_owned_dofs(), locally_relevant_F_dofs, mpi_comm);
  LinearAlgebra::distributed::Vector<double> F_rhs(F_partitioner);

  auto transaction =
    interaction_base.compute_projection_rhs_start("BSPLINE_3",
                                                  f_idx,
                                                  position_dof_handler,
                                                  position,
                                                  F_dof_handler,
                                                  F_mapping,
                                                  F_rhs);
  // This is necessary since InteractionBase isn't really intended to be used on
  // its own anyway
  auto &trans = dynamic_cast<fdl::Transaction<dim> &>(*transaction);
  trans.rhs_scatter_back_op = VectorOperation::add;

  transaction = interaction_base.compute_projection_rhs_intermediate(
    std::move(transaction));

  std::ofstream output;
  if (rank == 0)
    output.open("output");

  // plot overlap data on each processor:
  {
    // for the test we also need the overlap tria, which InteractionBase does
    // not make available. Set up our own (which should be equal) here:
    const auto patches =
      fdl::extract_patches(patch_hierarchy->getPatchLevel(level_number));
    const std::vector<BoundingBox<spacedim>> patch_bboxes =
      fdl::compute_patch_bboxes(patches, 1.0); // 1.0 needs to not be hard-coded
    fdl::BoxIntersectionPredicate<dim, spacedim> predicate(cell_bboxes,
                                                           patch_bboxes,
                                                           native_tria);
    fdl::OverlapTriangulation<dim> overlap_tria(native_tria, predicate);

    DoFHandler<dim> position_overlap_dof_handler(overlap_tria);
    position_overlap_dof_handler.distribute_dofs(position_fe);

    std::ostringstream this_output;
    if (input_db->getBoolWithDefault("write_mapped_cell_centers", false))
      {
        // TODO - implement this test to verify that the position scatter works.
        std::vector<types::global_dof_index> position_dofs(
          position_fe.dofs_per_cell);
        QMidpoint<dim>                           quadrature;
        MappingFEField<dim, dim, Vector<double>> position_map(
          position_overlap_dof_handler, trans.overlap_position);

        FEValues<dim> fe_values(position_map,
                                position_fe,
                                quadrature,
                                update_quadrature_points);

        for (const auto &cell :
             position_overlap_dof_handler.active_cell_iterators())
          {
            fe_values.reinit(cell);

            const auto center        = cell->center();
            const auto mapped_center = fe_values.get_quadrature_points()[0];

            this_output << "center = " << cell->center() << " mapped center = "
                        << fe_values.get_quadrature_points()[0]
                        << " equality = " << std::boolalpha
                        << (center.distance(mapped_center) < 1e-12) << '\n';
          }
      }

    print_strings_on_0(this_output.str(), mpi_comm, output);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(position_overlap_dof_handler);
    data_out.add_data_vector(trans.overlap_position, "position");
    data_out.build_patches();
    std::ofstream data_out_stream("overlap-tria-" + std::to_string(rank) +
                                  ".vtu");
    data_out.write_vtu(data_out_stream);
  }

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  // plot native solution:
  {
#if 0
    DataOut<dim> native_data_out;
    native_data_out.attach_dof_handler(F_dof_handler);
    native_solution.update_ghost_values();
    native_data_out.add_data_vector(native_solution, "F");

    Vector<float> subdomain(native_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = native_tria.locally_owned_subdomain();
    native_data_out.add_data_vector(subdomain, "subdomain");

    native_data_out.build_patches();

    native_data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_comm, 2, 8);
#endif
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
