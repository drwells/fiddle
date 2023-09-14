#include <fiddle/base/exceptions.h>
#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/interaction/nodal_interaction.h>

#include <fiddle/postprocess/meter_base.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <ibtk/IndexUtilities.h>

#include <PatchLevel.h>
#include <CartesianGridGeometry.h>
#include <tbox/InputManager.h>

#include <cmath>
#include <limits>

namespace fdl
{
  template <int dim, int spacedim>
  MeterBase<dim, spacedim>::MeterBase(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim, spacedim>>(1))
    , vector_fe(std::make_unique<FESystem<dim, spacedim>>(*scalar_fe, spacedim))
  {}

  template <int dim, int spacedim>
  MeterBase<dim, spacedim>::MeterBase(
    const Triangulation<dim, spacedim>           &tria,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
  {
    AssertThrow(!tria.has_hanging_nodes(), ExcFDLNotImplemented());
    GridGenerator::flatten_triangulation(tria, meter_tria);

    if (tria.all_reference_cells_are_simplex())
      scalar_fe = std::make_unique<FE_SimplexP<dim, spacedim>>(1);
    else if (tria.all_reference_cells_are_hyper_cube())
      scalar_fe = std::make_unique<FE_Q<dim, spacedim>>(1);
    else
      AssertThrow(false,
                  ExcMessage("mixed meshes are not yet supported here."));

    vector_fe = std::make_unique<FESystem<dim, spacedim>>(*scalar_fe, spacedim);
  }

  template <int dim, int spacedim>
  MeterBase<dim, spacedim>::~MeterBase()
  {}

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::reinit_dofs()
  {
    meter_mapping = meter_tria.get_reference_cells()[0]
                      .template get_default_mapping<dim, spacedim>(
                        scalar_fe->tensor_degree());
    if (meter_tria.all_reference_cells_are_simplex())
      meter_quadrature =
        QWitherdenVincentSimplex<dim>(scalar_fe->tensor_degree() + 1);
    else
      meter_quadrature = QGauss<dim>(scalar_fe->tensor_degree() + 1);

    scalar_dof_handler.reinit(meter_tria);
    scalar_dof_handler.distribute_dofs(*scalar_fe);
    vector_dof_handler.reinit(meter_tria);
    vector_dof_handler.distribute_dofs(*vector_fe);

    // Set up partitioners:
    const MPI_Comm comm = meter_tria.get_communicator();
    {
      IndexSet scalar_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(scalar_dof_handler,
                                              scalar_locally_relevant_dofs);
      scalar_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        scalar_dof_handler.locally_owned_dofs(),
        scalar_locally_relevant_dofs,
        comm);

      IndexSet vector_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(vector_dof_handler,
                                              vector_locally_relevant_dofs);
      vector_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        vector_dof_handler.locally_owned_dofs(),
        vector_locally_relevant_dofs,
        comm);
    }
    identity_position.reinit(vector_partitioner);
    VectorTools::interpolate(vector_dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             identity_position);
    identity_position.update_ghost_values();
  }

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::reinit_centroid()
  {
    // Since we support codim-1 meshes, the centroid (computed with the integral
    // formula) may not actually exist in the mesh. Find an equivalent point on
    // the mesh by
    // 1. Compute the analytic centroid.
    // 2. Find the cell closest to the analytic centroid. Increase the numerical
    //    tolerance until we find at least one cell.
    // 3. If there are multiple cells, canonicalize by picking the one with the
    //    lowest index.
    // 4. Broadcast the result.
    const MPI_Comm comm = get_triangulation().get_communicator();
    const int      rank = Utilities::MPI::this_mpi_process(comm);

    // Step 1
    Point<spacedim> a_centroid;
    for (unsigned int d = 0; d < spacedim; ++d)
      a_centroid[d] = VectorTools::compute_mean_value(get_mapping(),
                                                      get_vector_dof_handler(),
                                                      meter_quadrature,
                                                      identity_position,
                                                      d);
    std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
              Point<dim>>
      centroid_pair;
    // Step 2
    double       tolerance  = 1e-14;
    bool         found_cell = false;
    const double dx         = compute_min_cell_width(patch_hierarchy);
    do
      {
        centroid_pair = GridTools::find_active_cell_around_point(get_mapping(),
                                                                 meter_tria,
                                                                 a_centroid);
        // Ignore ghost cells
        if (centroid_pair.first != meter_tria.end() &&
            !centroid_pair.first->is_locally_owned())
          {
            Assert(centroid_pair.first->is_ghost(), ExcFDLInternalError());
            centroid_pair.first = meter_tria.end();
          }

        tolerance *= 2.0;
        // quit if at least one processor found the cell
        found_cell =
          Utilities::MPI::sum(int(centroid_pair.first != meter_tria.end()),
                              comm) != 0;
    } while (!found_cell && tolerance < dx);
    AssertThrow(found_cell, ExcFDLInternalError());

    // Step 3
    int index_rank[2]{std::numeric_limits<int>::max(), rank};
    if (centroid_pair.first != meter_tria.end())
      {
        AssertThrow(centroid_pair.first->level() == 0, ExcFDLNotImplemented());
        index_rank[0] = centroid_pair.first->index();
      }
    int result[2]{std::numeric_limits<int>::max(),
                  std::numeric_limits<int>::max()};
    int ierr =
      MPI_Allreduce(&index_rank, &result, 1, MPI_2INT, MPI_MINLOC, comm);
    AssertThrowMPI(ierr);
    Assert(result[0] != std::numeric_limits<int>::max(), ExcFDLInternalError());
    ref_centroid =
      Utilities::MPI::broadcast(comm, centroid_pair.second, result[1]);

    // Step 4
    centroid_cell = TriaActiveIterator<CellAccessor<dim, spacedim>>(&meter_tria,
                                                                    0,
                                                                    result[0],
                                                                    nullptr);
    if (centroid_cell->is_locally_owned())
      {
        Assert(int(centroid_cell->subdomain_id()) == rank,
               ExcFDLInternalError());
        Assert(result[1] == rank, ExcFDLInternalError());
        Assert(centroid_pair.first == centroid_cell, ExcFDLInternalError());
        centroid = get_mapping().transform_unit_to_real_cell(centroid_cell,
                                                             ref_centroid);
      }
    centroid = Utilities::MPI::broadcast(comm, centroid, result[1]);
  }

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::reinit_interaction()
  {
    // As the meter mesh is in absolute coordinates we can use a normal
    // mapping here
    const auto local_bboxes =
      compute_cell_bboxes<dim, spacedim, float>(get_vector_dof_handler(),
                                                get_mapping());
    const auto all_bboxes =
      collect_all_active_cell_bboxes(meter_tria, local_bboxes);

    // 1e-6 is an arbitrary nonzero number which ensures that points on the
    // boundaries between patches end up in both (for the purposes of
    // computing interpolations) but minimizes the amount of work resulting
    // from double-counting. I suspect that any number larger than 1e-10 would
    // suffice.
    tbox::Pointer<tbox::Database> db = new tbox::InputDatabase("meter_mesh_db");
    db->putDouble("ghost_cell_fraction", 1e-6);
    nodal_interaction = std::make_unique<NodalInteraction<dim, spacedim>>(
      db,
      meter_tria,
      all_bboxes,
      patch_hierarchy,
      std::make_pair(0, patch_hierarchy->getFinestLevelNumber()),
      get_vector_dof_handler(),
      identity_position);
    nodal_interaction->add_dof_handler(get_scalar_dof_handler());
  }

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::internal_reinit()
  {
    reinit_dofs();
    reinit_centroid();
    reinit_interaction();
  }

  template <int dim, int spacedim>
  double
  MeterBase<dim, spacedim>::compute_centroid_value(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    // TODO: this is pretty wasteful but we don't have infrastructure set up to
    // do single point evaluations right now - ultimately this will be added to
    // IBAMR.
    const auto interpolated_data =
      interpolate_scalar_field(data_idx, kernel_name);

    double value = 0.0;
    if (centroid_cell->is_locally_owned())
      {
        Quadrature<dim> centroid_quad(ref_centroid);
        const auto     &fe = get_scalar_dof_handler().get_fe();

        FEValues<dim, spacedim> fe_values(get_mapping(),
                                          fe,
                                          centroid_quad,
                                          update_values);
        fe_values.reinit(centroid_cell);
        const auto cell =
          typename DoFHandler<dim, spacedim>::active_cell_iterator(
            &meter_tria,
            centroid_cell->level(),
            centroid_cell->index(),
            &get_scalar_dof_handler());
        std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
        cell->get_dof_indices(cell_dofs);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          value +=
            fe_values.shape_value(i, 0) * interpolated_data[cell_dofs[i]];
      }

    const int owning_rank =
      meter_tria
        .get_true_subdomain_ids_of_cells()[centroid_cell->active_cell_index()];
    value = Utilities::MPI::broadcast(meter_tria.get_communicator(),
                                      value,
                                      owning_rank);
    return value;
  }

  template <int dim, int spacedim>
  double
  MeterBase<dim, spacedim>::compute_mean_value(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    const auto interpolated_data =
      interpolate_scalar_field(data_idx, kernel_name);

    return VectorTools::compute_mean_value(get_mapping(),
                                           get_scalar_dof_handler(),
                                           meter_quadrature,
                                           interpolated_data,
                                           0);
  }

  template <int dim, int spacedim>
  LinearAlgebra::distributed::Vector<double>
  MeterBase<dim, spacedim>::interpolate_scalar_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    LinearAlgebra::distributed::Vector<double> interpolated_data(
      scalar_partitioner);
    nodal_interaction->interpolate(kernel_name,
                                   data_idx,
                                   get_vector_dof_handler(),
                                   identity_position,
                                   get_scalar_dof_handler(),
                                   get_mapping(),
                                   interpolated_data);
    interpolated_data.update_ghost_values();

    return interpolated_data;
  }

  template <int dim, int spacedim>
  LinearAlgebra::distributed::Vector<double>
  MeterBase<dim, spacedim>::interpolate_vector_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    LinearAlgebra::distributed::Vector<double> interpolated_data(
      vector_partitioner);
    nodal_interaction->interpolate(kernel_name,
                                   data_idx,
                                   get_vector_dof_handler(),
                                   identity_position,
                                   get_vector_dof_handler(),
                                   get_mapping(),
                                   interpolated_data);
    interpolated_data.update_ghost_values();

    return interpolated_data;
  }

  template <int dim, int spacedim>
  bool
  MeterBase<dim, spacedim>::compute_vertices_inside_domain() const
  {
    tbox::Pointer<geom::CartesianGridGeometry<spacedim>> geom =
      patch_hierarchy->getGridGeometry();
    Assert(geom, ExcFDLInternalError());
    tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
      patch_hierarchy->getPatchLevel(0);
    Assert(patch_level, ExcFDLInternalError());

    bool vertices_inside_domain = true;
    for (const auto &vertex : get_triangulation().get_vertices())
      {
        const auto index =
          IBTK::IndexUtilities::getCellIndex(vertex,
                                             geom,
                                             hier::IntVector<spacedim>(1));
        vertices_inside_domain =
          vertices_inside_domain &&
          patch_level->getPhysicalDomain().contains(index);
      }

    return vertices_inside_domain;
  }

  template class MeterBase<NDIM - 1, NDIM>;
  template class MeterBase<NDIM, NDIM>;
} // namespace fdl
