#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/postprocess/surface_meter.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <CartesianPatchGeometry.h>
#include <tbox/InputManager.h>

#include <cmath>
#include <limits>

namespace fdl
{
  namespace internal
  {
    namespace
    {
      // avoid "defined but not used" warnings by using NDIM
#if NDIM == 2
      void
      setup_meter_tria(const std::vector<Point<2>>    &boundary_points,
                       Triangulation<1, 2>            &tria,
                       const Triangle::AdditionalData &additional_data)
      {
        Assert(boundary_points.size() > 1, ExcFDLInternalError());
        std::vector<CellData<1>> cell_data;
        std::vector<Point<2>>    vertices;

        vertices.push_back(boundary_points[0]);
        unsigned int last_vertex_n = 0;
        for (unsigned int boundary_points_point_n = 0;
             boundary_points_point_n < boundary_points.size() - 1;
             ++boundary_points_point_n)
          {
            const Point<2> left  = boundary_points[boundary_points_point_n];
            const Point<2> right = boundary_points[boundary_points_point_n + 1];
            const double   boundary_points_length = (left - right).norm();
            unsigned int   n_subcells             = 1;
            if (additional_data.place_additional_boundary_vertices)
              n_subcells = static_cast<unsigned int>(std::ceil(
                boundary_points_length / additional_data.target_element_area));
            for (unsigned int subcell_n = 0; subcell_n < n_subcells;
                 ++subcell_n)
              {
                cell_data.emplace_back();
                cell_data.back().vertices[0] = last_vertex_n;
                vertices.push_back(left + (right - left) *
                                            double(subcell_n + 1) /
                                            double(n_subcells));
                ++last_vertex_n;
                cell_data.back().vertices[1] = last_vertex_n;
              }
          }

        SubCellData sub_cell_data;
        if (vertices.size() > 2)
          {
            std::vector<unsigned int> all_vertices;
            GridTools::delete_duplicated_vertices(vertices,
                                                  cell_data,
                                                  sub_cell_data,
                                                  all_vertices);
          }
        GridTools::consistently_order_cells(cell_data);
        tria.create_triangulation(vertices, cell_data, sub_cell_data);
      }
#else
      void
      setup_meter_tria(const std::vector<Point<3>>    &boundary_points,
                       Triangulation<2, 3>            &tria,
                       const Triangle::AdditionalData &additional_data)
      {
        Assert(boundary_points.size() > 2, ExcFDLInternalError());

        // the input may be a parallel Triangulation, so copy back-and-forth
        Triangulation<2, 3> serial_tria;
        create_planar_triangulation(boundary_points,
                                    serial_tria,
                                    additional_data);
        fit_boundary_vertices(boundary_points, serial_tria);
        tria.clear();
        tria.copy_triangulation(serial_tria);
      }
#endif
    } // namespace
  }   // namespace internal

  template <int dim, int spacedim>
  SurfaceMeter<dim, spacedim>::SurfaceMeter(
    const Mapping<dim, spacedim>                     &mapping,
    const DoFHandler<dim, spacedim>                  &position_dof_handler,
    const std::vector<Point<spacedim>>               &boundary_points,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
    : mapping(&mapping)
    , position_dof_handler(&position_dof_handler)
    , patch_hierarchy(patch_hierarchy)
    , point_values(std::make_unique<PointValues<spacedim, dim, spacedim>>(
        mapping,
        position_dof_handler,
        boundary_points))
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim - 1, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim - 1, spacedim>>(1))
    , vector_fe(
        std::make_unique<FESystem<dim - 1, spacedim>>(*scalar_fe, spacedim))
  {
    // TODO: assert congruity between position_dof_handler.get_communicator()
    // and SAMRAI_MPI::getCommunicator()
    reinit(position, velocity);
  }

  template <int dim, int spacedim>
  SurfaceMeter<dim, spacedim>::SurfaceMeter(
    const std::vector<Point<spacedim>>               &boundary_points,
    const std::vector<Tensor<1, spacedim>>           &velocity,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim - 1, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim - 1, spacedim>>(1))
    , vector_fe(
        std::make_unique<FESystem<dim - 1, spacedim>>(*scalar_fe, spacedim))
  {
    reinit(boundary_points, velocity);
  }

  template <int dim, int spacedim>
  bool
  SurfaceMeter<dim, spacedim>::uses_codim_zero_mesh() const
  {
    return position_dof_handler != nullptr;
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit(
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
  {
    Assert(uses_codim_zero_mesh(),
           ExcMessage("This function cannot be called when the SurfaceMeter is "
                      "set up without an underlying codimension zero "
                      "Triangulation."));
    // Reset the meter mesh according to the new position values:
    const std::vector<Tensor<1, spacedim>> position_values =
      point_values->evaluate(position);
    const std::vector<Point<spacedim>> boundary_points(position_values.begin(),
                                                       position_values.end());
    const std::vector<Tensor<1, spacedim>> velocity_values =
      point_values->evaluate(velocity);

    reinit_tria(boundary_points, false);
    reinit_mean_velocity(velocity_values);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit(
    const std::vector<Point<spacedim>>     &boundary_points,
    const std::vector<Tensor<1, spacedim>> &velocity_values)
  {
    Assert(!uses_codim_zero_mesh(),
           ExcMessage("This function may only be called when the SurfaceMeter "
                      "is set up without an underlying codimension zero "
                      "Triangulation."));
    reinit_tria(boundary_points, true);
    reinit_mean_velocity(velocity_values);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit_tria(
    const std::vector<Point<spacedim>> &boundary_points,
    const bool                          place_additional_boundary_vertices)
  {
    double dx_0 = std::numeric_limits<double>::max();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber());
    Assert(level, ExcFDLInternalError());
    for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
      {
        tbox::Pointer<hier::Patch<spacedim>> patch = level->getPatch(p());
        const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> pgeom =
          patch->getPatchGeometry();
        dx_0 = std::min(dx_0,
                        *std::min_element(pgeom->getDx(),
                                          pgeom->getDx() + spacedim));
      }
    dx_0 = Utilities::MPI::min(dx_0, tbox::SAMRAI_MPI::getCommunicator());
    Assert(dx_0 != std::numeric_limits<double>::max(), ExcFDLInternalError());
    const double target_element_area = std::pow(dx_0, dim - 1);

    meter_tria.clear();
    Triangle::AdditionalData additional_data;
    additional_data.target_element_area = target_element_area;
    additional_data.place_additional_boundary_vertices =
      place_additional_boundary_vertices;
    internal::setup_meter_tria(boundary_points, meter_tria, additional_data);

    meter_mapping = meter_tria.get_reference_cells()[0]
                      .template get_default_mapping<dim - 1, spacedim>(
                        scalar_fe->tensor_degree());
    meter_quadrature =
      QWitherdenVincentSimplex<dim - 1>(scalar_fe->tensor_degree() + 1);

    scalar_dof_handler.reinit(meter_tria);
    scalar_dof_handler.distribute_dofs(*scalar_fe);
    vector_dof_handler.reinit(meter_tria);
    vector_dof_handler.distribute_dofs(*vector_fe);

    // As the meter mesh is in absolute coordinates we can use a normal
    // mapping here
    const auto local_bboxes =
      compute_cell_bboxes<dim - 1, spacedim, float>(vector_dof_handler,
                                                    *meter_mapping);
    const auto all_bboxes =
      collect_all_active_cell_bboxes(meter_tria, local_bboxes);

    // Set up partitioners:
    const MPI_Comm comm = meter_tria.get_communicator();
    const int      rank = Utilities::MPI::this_mpi_process(comm);
    {
      IndexSet vector_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(vector_dof_handler,
                                              vector_locally_relevant_dofs);
      vector_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        vector_dof_handler.locally_owned_dofs(),
        vector_locally_relevant_dofs,
        comm);

      IndexSet scalar_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(scalar_dof_handler,
                                              scalar_locally_relevant_dofs);
      scalar_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        scalar_dof_handler.locally_owned_dofs(),
        scalar_locally_relevant_dofs,
        comm);
    }
    identity_position.reinit(vector_partitioner);
    VectorTools::interpolate(vector_dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             identity_position);
    identity_position.update_ghost_values();

    // Since we have codim-1 meshes, the centroid (computed with the integral
    // formula) may not actually exist in the mesh. Find an equivalent point on
    // the mesh by
    // 1. Compute the analytic centroid.
    // 2. Find the cell closest to the analytic centroid. Increase the numerical
    //    tolerance until we find at least one cell.
    // 3. If there are multiple cells, canonicalize by picking the one with the
    //    lowest index.
    // 4. Broadcast the result.

    // Step 1
    Point<spacedim> a_centroid;
    for (unsigned int d = 0; d < spacedim; ++d)
      a_centroid[d] = VectorTools::compute_mean_value(get_mapping(),
                                                      get_vector_dof_handler(),
                                                      meter_quadrature,
                                                      identity_position,
                                                      d);
    std::pair<typename Triangulation<dim - 1, spacedim>::active_cell_iterator,
              Point<dim - 1>>
      centroid_pair;
    // Step 2
    double tolerance  = 1e-14;
    bool   found_cell = false;
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
    } while (!found_cell && tolerance < dx_0);
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
    centroid_cell = TriaActiveIterator<CellAccessor<dim - 1, spacedim>>(
      &meter_tria, 0, result[0], nullptr);
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

    // 1e-6 is an arbitrary nonzero number which ensures that points on the
    // boundaries between patches end up in both (for the purposes of
    // computing interpolations) but minimizes the amount of work resulting
    // from double-counting. I suspect that any number larger than 1e-10 would
    // suffice.
    tbox::Pointer<tbox::Database> db = new tbox::InputDatabase("meter_mesh_db");
    db->putDouble("ghost_cell_fraction", 1e-6);
    nodal_interaction = std::make_unique<NodalInteraction<dim - 1, spacedim>>(
      db,
      meter_tria,
      all_bboxes,
      patch_hierarchy,
      std::make_pair(0, patch_hierarchy->getFinestLevelNumber()),
      vector_dof_handler,
      identity_position);
    nodal_interaction->add_dof_handler(scalar_dof_handler);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit_mean_velocity(
    const std::vector<Tensor<1, spacedim>> &velocity_values)
  {
    if (dim == 2)
      {
        // Average the velocities (there should only be two anyway).
        mean_velocity = std::accumulate(velocity_values.begin(),
                                        velocity_values.end(),
                                        Tensor<1, spacedim>()) *
                        (1.0 / double(velocity_values.size()));
      }
    if (dim == 3)
      {
        // Avoid funky linker errors in 2D by manually implementing the
        // trapezoid rule
        std::vector<Point<dim - 2>> points;
        points.emplace_back(0.0);
        points.emplace_back(1.0);
        std::vector<double> weights;
        weights.emplace_back(0.5);
        weights.emplace_back(0.5);
        Quadrature<dim - 2>           face_quadrature(points, weights);
        FE_Nothing<dim - 1, spacedim> fe_nothing(
          meter_tria.get_reference_cells()[0]);
        FEFaceValues<dim - 1, spacedim> face_values(get_mapping(),
                                                    fe_nothing,
                                                    face_quadrature,
                                                    update_JxW_values);
        mean_velocity                 = 0.0;
        double       area             = 0.0;
        unsigned int n_boundary_faces = 0;
        for (const auto &cell : meter_tria.active_cell_iterators())
          for (unsigned int face_no : cell->face_indices())
            if (cell->face(face_no)->at_boundary())
              {
                face_values.reinit(cell, face_no);
                const auto f = cell->face(face_no);
                AssertIndexRange(f->vertex_index(0), velocity_values.size());
                AssertIndexRange(f->vertex_index(1), velocity_values.size());
                const auto v0   = velocity_values[f->vertex_index(0)];
                const auto v1   = velocity_values[f->vertex_index(1)];
                const auto JxW0 = face_values.get_JxW_values()[0];
                const auto JxW1 = face_values.get_JxW_values()[1];

                mean_velocity += v0 * JxW0;
                mean_velocity += v1 * JxW1;
                area += JxW0;
                area += JxW1;
                ++n_boundary_faces;
              }
        mean_velocity *= 1.0 / area;
        AssertThrow(n_boundary_faces == velocity_values.size(),
                    ExcMessage("There should be exactly one boundary face for "
                               "every boundary vertex, and one velocity value "
                               "for each boundary vertex."));
      }
  }

  template <int dim, int spacedim>
  LinearAlgebra::distributed::Vector<double>
  SurfaceMeter<dim, spacedim>::interpolate_scalar_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    LinearAlgebra::distributed::Vector<double> interpolated_data(
      scalar_partitioner);
    auto transaction = nodal_interaction->compute_projection_rhs_forward_start(
      kernel_name,
      data_idx,
      vector_dof_handler,
      identity_position,
      scalar_dof_handler,
      *meter_mapping,
      interpolated_data);
    transaction = nodal_interaction->compute_projection_rhs_forward_finish(
      std::move(transaction));
    transaction = nodal_interaction->compute_projection_rhs_intermediate(
      std::move(transaction));
    nodal_interaction->compute_projection_rhs_finish(std::move(transaction));
    interpolated_data.update_ghost_values();

    return interpolated_data;
  }

  template <int dim, int spacedim>
  LinearAlgebra::distributed::Vector<double>
  SurfaceMeter<dim, spacedim>::interpolate_vector_field(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    LinearAlgebra::distributed::Vector<double> interpolated_data(
      vector_partitioner);
    auto transaction = nodal_interaction->compute_projection_rhs_forward_start(
      kernel_name,
      data_idx,
      vector_dof_handler,
      identity_position,
      vector_dof_handler,
      *meter_mapping,
      interpolated_data);
    transaction = nodal_interaction->compute_projection_rhs_forward_finish(
      std::move(transaction));
    transaction = nodal_interaction->compute_projection_rhs_intermediate(
      std::move(transaction));
    nodal_interaction->compute_projection_rhs_finish(std::move(transaction));
    interpolated_data.update_ghost_values();

    return interpolated_data;
  }

  template <int dim, int spacedim>
  double
  SurfaceMeter<dim, spacedim>::compute_mean_value(
    const int          data_idx,
    const std::string &kernel_name)
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
  double
  SurfaceMeter<dim, spacedim>::compute_flux(const int          data_idx,
                                            const std::string &kernel_name)
  {
    const auto interpolated_data =
      interpolate_vector_field(data_idx, kernel_name);

    const auto                 &fe = get_vector_dof_handler().get_fe();
    FEValues<dim - 1, spacedim> fe_values(get_mapping(),
                                          fe,
                                          meter_quadrature,
                                          update_normal_vectors |
                                            update_values | update_JxW_values);

    std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
    std::vector<Tensor<1, spacedim>>     cell_values(meter_quadrature.size());
    double                               flux = 0.0;
    for (const auto &cell : get_vector_dof_handler().active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
      {
        cell->get_dof_indices(cell_dofs);
        fe_values.reinit(cell);
        fe_values[FEValuesExtractors::Vector(0)].get_function_values(
          interpolated_data, cell_values);
        for (unsigned int q = 0; q < meter_quadrature.size(); ++q)
          flux +=
            cell_values[q] * fe_values.normal_vector(q) * fe_values.JxW(q);
      }

    return Utilities::MPI::sum(flux, meter_tria.get_communicator());
  }

  template <int dim, int spacedim>
  double
  SurfaceMeter<dim, spacedim>::compute_centroid_value(
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
        Quadrature<dim - 1> centroid_quad(ref_centroid);
        const auto         &fe = get_scalar_dof_handler().get_fe();

        FEValues<dim - 1, spacedim> fe_values(get_mapping(),
                                              fe,
                                              centroid_quad,
                                              update_values);
        fe_values.reinit(centroid_cell);
        const auto cell =
          typename DoFHandler<dim - 1, spacedim>::active_cell_iterator(
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

  template class SurfaceMeter<NDIM, NDIM>;
} // namespace fdl
