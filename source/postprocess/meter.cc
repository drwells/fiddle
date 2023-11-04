#include <fiddle/base/exceptions.h>
#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/interaction/nodal_interaction.h>

#include <fiddle/postprocess/meter.h>

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

#include <deal.II/numerics/vector_tools_mean_value.h>

#include <ibtk/IndexUtilities.h>

#include <CartesianGridGeometry.h>
#include <PatchLevel.h>
#include <tbox/InputManager.h>

#include <cmath>
#include <limits>

namespace fdl
{
  template <int dim, int spacedim>
  Meter<dim, spacedim>::Meter(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
  {}

  template <int dim, int spacedim>
  Meter<dim, spacedim>::Meter(
    const Triangulation<dim, spacedim>           &tria,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_ctor, "fdl::Meter::Meter()");
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
  Meter<dim, spacedim>::~Meter()
  {}

  template <int dim, int spacedim>
  void
  Meter<dim, spacedim>::reinit_dofs()
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_reinit_dofs, "fdl::Meter::reinit_dofs()");
    Assert(meter_tria.get_reference_cells().size() == 1,
           ExcFDLNotImplemented());
    // only set up FEs once
    if (!scalar_fe)
      {
        if (meter_tria.all_reference_cells_are_simplex())
          scalar_fe = std::make_unique<FE_SimplexP<dim, spacedim>>(1);
        else
          scalar_fe = std::make_unique<FE_Q<dim, spacedim>>(1);
        vector_fe =
          std::make_unique<FESystem<dim, spacedim>>(*scalar_fe, spacedim);
      }
    meter_mapping = meter_tria.get_reference_cells()[0]
                      .template get_default_mapping<dim, spacedim>(
                        scalar_fe->tensor_degree());
    // Since we have a faceted geometry with simplicies (i.e., the Jacobian on
    // each cell is constant) we can get away with using one degree lower
    if (meter_tria.all_reference_cells_are_simplex())
      meter_quadrature =
        QWitherdenVincentSimplex<dim>(scalar_fe->tensor_degree());
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

    // Directly calculate DoF locations. This is orders of magnitude faster than
    // VectorTools::interpolate().
    Assert(vector_fe->tensor_degree() == 1, ExcFDLNotImplemented());
    for (const auto &cell : get_vector_dof_handler().active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
      for (unsigned int vertex_no : cell->vertex_indices())
        {
          const Point<spacedim> vertex = cell->vertex(vertex_no);
          for (unsigned int d = 0; d < spacedim; ++d)
            identity_position[cell->vertex_dof_index(vertex_no, d)] = vertex[d];
        }

    identity_position.update_ghost_values();
  }

  template <int dim, int spacedim>
  void
  Meter<dim, spacedim>::reinit_centroid()
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_centroid,
                              "fdl::Meter::reinit_centroid()");
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
        centroid_pair = GridTools::find_active_cell_around_point(
          get_mapping(), meter_tria, a_centroid, {}, tolerance);
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
    //
    // Since we need the level, index, and rank, be a little lazy and scatter
    // them separately.
    int local_level_rank[2]{std::numeric_limits<int>::max(), rank};
    int local_index_rank[2]{std::numeric_limits<int>::max(), rank};
    if (centroid_pair.first != meter_tria.end())
      {
        local_level_rank[0] = centroid_pair.first->level();
        local_index_rank[0] = centroid_pair.first->index();
      }
    int level_rank[2]{std::numeric_limits<int>::max(),
                      std::numeric_limits<int>::max()};
    int index_rank[2]{std::numeric_limits<int>::max(),
                      std::numeric_limits<int>::max()};
    int ierr = MPI_Allreduce(
      &local_level_rank, &level_rank, 1, MPI_2INT, MPI_MINLOC, comm);
    AssertThrowMPI(ierr);
    ierr = MPI_Allreduce(
      &local_index_rank, &index_rank, 1, MPI_2INT, MPI_MINLOC, comm);
    AssertThrowMPI(ierr);
    Assert(level_rank[0] != std::numeric_limits<int>::max(),
           ExcFDLInternalError());
    Assert(index_rank[0] != std::numeric_limits<int>::max(),
           ExcFDLInternalError());
    ref_centroid =
      Utilities::MPI::broadcast(comm, centroid_pair.second, index_rank[1]);

    // Step 4
    centroid_cell = TriaActiveIterator<CellAccessor<dim, spacedim>>(
      &meter_tria, level_rank[0], index_rank[0], nullptr);
    for (unsigned int d = 0; d < spacedim; ++d)
      centroid[d] = std::numeric_limits<double>::signaling_NaN();
    if (centroid_cell->is_locally_owned())
      {
        Assert(int(centroid_cell->subdomain_id()) == rank,
               ExcFDLInternalError());
        Assert(index_rank[1] == rank, ExcFDLInternalError());
        Assert(centroid_pair.first == centroid_cell, ExcFDLInternalError());
        centroid = get_mapping().transform_unit_to_real_cell(centroid_cell,
                                                             ref_centroid);
      }
    centroid = Utilities::MPI::broadcast(comm, centroid, index_rank[1]);
    Assert(!std::isnan(centroid[0]), ExcFDLInternalError());
  }

  template <int dim, int spacedim>
  void
  Meter<dim, spacedim>::reinit_interaction()
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_reinit_interaction,
                              "fdl::Meter::reinit_interaction()");
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
  Meter<dim, spacedim>::internal_reinit()
  {
    reinit_dofs();
    reinit_centroid();
    reinit_interaction();
  }

  template <int dim, int spacedim>
  double
  Meter<dim, spacedim>::compute_centroid_value(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_centroid_value,
                              "fdl::Meter::compute_centroid_value()");
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
  Meter<dim, spacedim>::compute_mean_value(const int          data_idx,
                                           const std::string &kernel_name) const
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_mean_value,
                              "fdl::Meter::compute_mean_value()");
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
  Meter<dim, spacedim>::interpolate_scalar_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_interpolate_scalar,
                              "fdl::Meter::interpolate_scalar_field()");
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
  Meter<dim, spacedim>::interpolate_vector_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_interpolate_vector,
                              "fdl::Meter::interpolate_vector_field()");
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
  Meter<dim, spacedim>::compute_vertices_inside_domain() const
  {
    FDL_SETUP_TIMER_AND_SCOPE(t_meter_vertices_inside_domain,
                              "fdl::Meter::compute_vertices_inside_domain()");
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

  template class Meter<NDIM - 1, NDIM>;
  template class Meter<NDIM, NDIM>;
} // namespace fdl
