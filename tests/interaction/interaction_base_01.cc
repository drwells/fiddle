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

#include <fiddle/base/exceptions.h>

#include <fstream>

#include "../tests.h"

using namespace SAMRAI;
using namespace dealii;

// First test driver for InteractionBase. Used to test
//
// 1. quadrature index scattering

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
  auto pair            = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = pair.first;
  auto u_cc_idx        = pair.second;

  // Now set up fiddle things for the test:
  std::vector<BoundingBox<spacedim, float>> cell_bboxes;
  for (const auto cell : native_tria.active_cell_iterators())
    {
      BoundingBox<spacedim, float> fbbox;
      fbbox.get_boundary_points() = cell->bounding_box().get_boundary_points();
      cell_bboxes.push_back(fbbox);
    }

  auto eulerian_data_cache = std::make_shared<IBTK::SAMRAIDataCache>();
  eulerian_data_cache->setPatchHierarchy(patch_hierarchy);
  const auto level_number = patch_hierarchy->getFinestLevelNumber();
  eulerian_data_cache->resetLevels(level_number, level_number);
  fdl::InteractionBase<dim, spacedim> interaction_base(
    native_tria,
    cell_bboxes,
    patch_hierarchy,
    patch_hierarchy->getFinestLevelNumber(),
    eulerian_data_cache);

  const fdl::SingleQuadrature<dim> single_quad(QGauss<dim>(2));
  // for this test set the cell index
  std::vector<unsigned char> quadrature_indices(
    native_tria.n_locally_owned_active_cells());
  unsigned int cell_n = 0;
  for (const auto &cell : native_tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const double x   = cell->barycenter()[0];
        const double sgn = std::signbit(x) == 0 ? 1.0 : -1.0;
        quadrature_indices[cell_n] =
          100 + sgn * std::min(100.0, 100.0 * std::abs(x));
        ++cell_n;
      }

  FESystem<dim>             X_fe(FE_Q<dim>(1), dim);
  DoFHandler<dim, spacedim> X_dof_handler(native_tria);
  X_dof_handler.distribute_dofs(X_fe);
  interaction_base.add_dof_handler(X_dof_handler);

  IndexSet locally_relevant_X_dofs;
  DoFTools::extract_locally_relevant_dofs(X_dof_handler,
                                          locally_relevant_X_dofs);
  auto X_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    X_dof_handler.locally_owned_dofs(), locally_relevant_X_dofs, mpi_comm);
  LinearAlgebra::distributed::Vector<double> X(X_partitioner);
  VectorTools::interpolate(X_dof_handler, Identity<dim>(), X);
  X.update_ghost_values();

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
    interaction_base.compute_projection_rhs_start(u_cc_idx,
                                                  single_quad,
                                                  quadrature_indices,
                                                  X_dof_handler,
                                                  X,
                                                  F_dof_handler,
                                                  F_mapping,
                                                  F_rhs);

  transaction = interaction_base.compute_projection_rhs_intermediate(
    std::move(transaction));
  auto &trans = dynamic_cast<fdl::Transaction<dim> &>(*transaction);

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

    DoFHandler<dim> X_overlap_dof_handler(overlap_tria);
    X_overlap_dof_handler.distribute_dofs(X_fe);
    Vector<float> quad_indices(overlap_tria.n_active_cells());
    Assert(quad_indices.size() == trans.overlap_quad_indices.size(),
           ExcMessage("wrong size overlap quad indices"));
    std::copy(trans.overlap_quad_indices.begin(),
              trans.overlap_quad_indices.end(),
              quad_indices.begin());

    std::ostringstream this_output;
    if (input_db->getBoolWithDefault("write_quad_indices", false))
      {
        this_output << "rank = " << rank << '\n';
        for (const auto cell : overlap_tria.active_cell_iterators())
          {
            this_output << "barycenter = " << cell->barycenter()
                        << " quad index = "
                        << quad_indices[cell->active_cell_index()] << '\n';
          }

        print_strings_on_0(this_output.str(), output);
      }

    if (input_db->getBoolWithDefault("write_mapped_cell_centers", false))
      {
        // TODO - implement this test to verify that the X scatter works.
      }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(X_overlap_dof_handler);
    data_out.add_data_vector(quad_indices, "quadrature_indices");
    data_out.add_data_vector(trans.overlap_X_vec, "position");
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
#if 0 // TODO plot quad indices
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
