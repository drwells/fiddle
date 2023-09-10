#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/postprocess/surface_meter.h>

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

#include <CartesianPatchGeometry.h>

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

      template <int spacedim>
      double
      compute_min_cell_width(
        const tbox::Pointer<hier::PatchHierarchy<spacedim>> &patch_hierarchy)
      {
        double dx = std::numeric_limits<double>::max();
        const tbox::Pointer<hier::PatchLevel<spacedim>> level =
          patch_hierarchy->getPatchLevel(
            patch_hierarchy->getFinestLevelNumber());
        Assert(level, ExcFDLInternalError());
        for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
          {
            const tbox::Pointer<hier::Patch<spacedim>> patch =
              level->getPatch(p());
            const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> pgeom =
              patch->getPatchGeometry();
            dx = std::min(dx,
                          *std::min_element(pgeom->getDx(),
                                            pgeom->getDx() + spacedim));
          }
        dx = Utilities::MPI::min(dx, tbox::SAMRAI_MPI::getCommunicator());
        Assert(dx != std::numeric_limits<double>::max(), ExcFDLInternalError());
        return dx;
      }
    } // namespace
  }   // namespace internal

  template <int dim, int spacedim>
  SurfaceMeter<dim, spacedim>::SurfaceMeter(
    const Mapping<dim, spacedim>                     &mapping,
    const DoFHandler<dim, spacedim>                  &position_dof_handler,
    const std::vector<Point<spacedim>>               &boundary_points,
    tbox::Pointer<hier::PatchHierarchy<spacedim>>     patch_hierarchy,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
    : MeterBase<dim - 1, spacedim>(patch_hierarchy)
    , mapping(&mapping)
    , position_dof_handler(&position_dof_handler)
    , point_values(std::make_unique<PointValues<spacedim, dim, spacedim>>(
        mapping,
        position_dof_handler,
        boundary_points))
  {
    // TODO: assert congruity between position_dof_handler.get_communicator()
    // and SAMRAI_MPI::getCommunicator()
    reinit(position, velocity);
  }

  template <int dim, int spacedim>
  SurfaceMeter<dim, spacedim>::SurfaceMeter(
    const Triangulation<dim - 1, spacedim>       &tria,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : MeterBase<dim - 1, spacedim>(tria, patch_hierarchy)
  {
    internal_reinit(false, {}, {}, false);
  }

  template <int dim, int spacedim>
  SurfaceMeter<dim, spacedim>::SurfaceMeter(
    const std::vector<Point<spacedim>>           &boundary_points,
    const std::vector<Tensor<1, spacedim>>       &velocity,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : MeterBase<dim - 1, spacedim>(patch_hierarchy)
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

    internal_reinit(true, boundary_points, velocity_values, false);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit(
    const std::vector<Point<spacedim>>     &boundary_points,
    const std::vector<Tensor<1, spacedim>> &velocity_values)
  {
    AssertThrow(!uses_codim_zero_mesh(),
                ExcMessage("This function may only be called when the "
                           "SurfaceMeter is set up without an underlying "
                           "codimension zero Triangulation."));
    internal_reinit(true, boundary_points, velocity_values, spacedim != 3);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit()
  {
    AssertThrow(!uses_codim_zero_mesh(),
                ExcMessage("This function may only be called when the "
                           "SurfaceMeter is set up without an underlying "
                           "codimension zero Triangulation."));
    // special case: nothing can move so skip all but one reinit function
    this->reinit_interaction();
  }


  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit_tria(
    const std::vector<Point<spacedim>> &boundary_points,
    const bool                          place_additional_boundary_vertices)
  {
    const double dx = internal::compute_min_cell_width(this->patch_hierarchy);
    const double target_element_area = std::pow(dx, dim - 1);

    this->meter_tria.clear();
    Triangle::AdditionalData additional_data;
    additional_data.target_element_area = target_element_area;
    additional_data.place_additional_boundary_vertices =
      place_additional_boundary_vertices;
    internal::setup_meter_tria(boundary_points,
                               this->meter_tria,
                               additional_data);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit_centroid()
  {
    // Since we have codim-1 meshes, the centroid (computed with the integral
    // formula) may not actually exist in the mesh. Find an equivalent point on
    // the mesh by
    // 1. Compute the analytic centroid.
    // 2. Find the cell closest to the analytic centroid. Increase the numerical
    //    tolerance until we find at least one cell.
    // 3. If there are multiple cells, canonicalize by picking the one with the
    //    lowest index.
    // 4. Broadcast the result.
    const MPI_Comm comm = this->meter_tria.get_communicator();
    const int      rank = Utilities::MPI::this_mpi_process(comm);

    // Step 1
    Point<spacedim> a_centroid;
    for (unsigned int d = 0; d < spacedim; ++d)
      a_centroid[d] =
        VectorTools::compute_mean_value(this->get_mapping(),
                                        this->get_vector_dof_handler(),
                                        this->meter_quadrature,
                                        this->identity_position,
                                        d);
    std::pair<typename Triangulation<dim - 1, spacedim>::active_cell_iterator,
              Point<dim - 1>>
      centroid_pair;
    // Step 2
    double       tolerance  = 1e-14;
    bool         found_cell = false;
    const double dx = internal::compute_min_cell_width(this->patch_hierarchy);
    do
      {
        centroid_pair =
          GridTools::find_active_cell_around_point(this->get_mapping(),
                                                   this->meter_tria,
                                                   a_centroid);
        // Ignore ghost cells
        if (centroid_pair.first != this->meter_tria.end() &&
            !centroid_pair.first->is_locally_owned())
          {
            Assert(centroid_pair.first->is_ghost(), ExcFDLInternalError());
            centroid_pair.first = this->meter_tria.end();
          }

        tolerance *= 2.0;
        // quit if at least one processor found the cell
        found_cell = Utilities::MPI::sum(int(centroid_pair.first !=
                                             this->meter_tria.end()),
                                         comm) != 0;
    } while (!found_cell && tolerance < dx);
    AssertThrow(found_cell, ExcFDLInternalError());

    // Step 3
    int index_rank[2]{std::numeric_limits<int>::max(), rank};
    if (centroid_pair.first != this->meter_tria.end())
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
      &this->meter_tria, 0, result[0], nullptr);
    if (centroid_cell->is_locally_owned())
      {
        Assert(int(centroid_cell->subdomain_id()) == rank,
               ExcFDLInternalError());
        Assert(result[1] == rank, ExcFDLInternalError());
        Assert(centroid_pair.first == centroid_cell, ExcFDLInternalError());
        centroid =
          this->get_mapping().transform_unit_to_real_cell(centroid_cell,
                                                          ref_centroid);
      }
    centroid = Utilities::MPI::broadcast(comm, centroid, result[1]);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::internal_reinit(
    const bool                              reinit_tria,
    const std::vector<Point<spacedim>>     &boundary_points,
    const std::vector<Tensor<1, spacedim>> &velocity_values,
    const bool                              place_additional_boundary_vertices)
  {
    if (reinit_tria)
      this->reinit_tria(boundary_points, place_additional_boundary_vertices);
    this->reinit_dofs();
    reinit_centroid();
    this->reinit_interaction();
    reinit_mean_velocity(velocity_values);
  }

  template <int dim, int spacedim>
  void
  SurfaceMeter<dim, spacedim>::reinit_mean_velocity(
    const std::vector<Tensor<1, spacedim>> &velocity_values)
  {
    if (velocity_values.size() == 0)
      {
        mean_velocity = 0.0;
        return;
      }

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
          this->meter_tria.get_reference_cells()[0]);
        FEFaceValues<dim - 1, spacedim> face_values(this->get_mapping(),
                                                    fe_nothing,
                                                    face_quadrature,
                                                    update_JxW_values);
        mean_velocity                 = 0.0;
        double       area             = 0.0;
        unsigned int n_boundary_faces = 0;
        for (const auto &cell : this->meter_tria.active_cell_iterators())
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
      this->scalar_partitioner);
    this->nodal_interaction->interpolate(kernel_name,
                                         data_idx,
                                         this->get_vector_dof_handler(),
                                         this->identity_position,
                                         this->get_scalar_dof_handler(),
                                         this->get_mapping(),
                                         interpolated_data);
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
      this->vector_partitioner);
    this->nodal_interaction->interpolate(kernel_name,
                                         data_idx,
                                         this->get_vector_dof_handler(),
                                         this->identity_position,
                                         this->get_vector_dof_handler(),
                                         this->get_mapping(),
                                         interpolated_data);
    interpolated_data.update_ghost_values();

    return interpolated_data;
  }

  template <int dim, int spacedim>
  double
  SurfaceMeter<dim, spacedim>::compute_mean_value(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    const auto interpolated_data =
      interpolate_scalar_field(data_idx, kernel_name);

    return VectorTools::compute_mean_value(this->get_mapping(),
                                           this->get_scalar_dof_handler(),
                                           this->meter_quadrature,
                                           interpolated_data,
                                           0);
  }

  template <int dim, int spacedim>
  std::pair<double, Tensor<1, spacedim>>
  SurfaceMeter<dim, spacedim>::compute_flux(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    const auto interpolated_data =
      interpolate_vector_field(data_idx, kernel_name);

    const auto                 &fe = this->get_vector_dof_handler().get_fe();
    FEValues<dim - 1, spacedim> fe_values(this->get_mapping(),
                                          fe,
                                          this->meter_quadrature,
                                          update_normal_vectors |
                                            update_values | update_JxW_values);

    std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
    std::vector<Tensor<1, spacedim>> cell_values(this->meter_quadrature.size());
    double                           flux = 0.0;
    Tensor<1, spacedim>              mean_normal;
    for (const auto &cell :
         this->get_vector_dof_handler().active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell())
      {
        cell->get_dof_indices(cell_dofs);
        fe_values.reinit(cell);
        fe_values[FEValuesExtractors::Vector(0)].get_function_values(
          interpolated_data, cell_values);
        for (unsigned int q = 0; q < this->meter_quadrature.size(); ++q)
          {
            flux +=
              cell_values[q] * fe_values.normal_vector(q) * fe_values.JxW(q);
            mean_normal += fe_values.normal_vector(q) * fe_values.JxW(q);
          }
      }

    flux = Utilities::MPI::sum(flux, this->meter_tria.get_communicator());
    mean_normal =
      Utilities::MPI::sum(mean_normal, this->meter_tria.get_communicator());
    mean_normal /= mean_normal.norm();
    return std::make_pair(flux, mean_normal);
  }

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  SurfaceMeter<dim, spacedim>::compute_mean_normal_vector() const
  {
    const auto                 &fe = this->get_vector_dof_handler().get_fe();
    FEValues<dim - 1, spacedim> fe_values(this->get_mapping(),
                                          fe,
                                          this->meter_quadrature,
                                          update_normal_vectors |
                                            update_values | update_JxW_values);

    Tensor<1, spacedim> mean_normal;
    for (const auto &cell :
         this->get_vector_dof_handler().active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell())
      {
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < this->meter_quadrature.size(); ++q)
          mean_normal += fe_values.normal_vector(q) * fe_values.JxW(q);
      }

    mean_normal =
      Utilities::MPI::sum(mean_normal, this->meter_tria.get_communicator());
    mean_normal /= mean_normal.norm();
    return mean_normal;
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
        const auto         &fe = this->get_scalar_dof_handler().get_fe();

        FEValues<dim - 1, spacedim> fe_values(this->get_mapping(),
                                              fe,
                                              centroid_quad,
                                              update_values);
        fe_values.reinit(centroid_cell);
        const auto cell =
          typename DoFHandler<dim - 1, spacedim>::active_cell_iterator(
            &this->meter_tria,
            centroid_cell->level(),
            centroid_cell->index(),
            &this->get_scalar_dof_handler());
        std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
        cell->get_dof_indices(cell_dofs);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          value +=
            fe_values.shape_value(i, 0) * interpolated_data[cell_dofs[i]];
      }

    const int owning_rank =
      this->meter_tria
        .get_true_subdomain_ids_of_cells()[centroid_cell->active_cell_index()];
    value = Utilities::MPI::broadcast(this->meter_tria.get_communicator(),
                                      value,
                                      owning_rank);
    return value;
  }

  template class SurfaceMeter<NDIM, NDIM>;
} // namespace fdl
