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
  SurfaceMeter<dim, spacedim>::internal_reinit(
    const bool                              reinit_tria,
    const std::vector<Point<spacedim>>     &boundary_points,
    const std::vector<Tensor<1, spacedim>> &velocity_values,
    const bool                              place_additional_boundary_vertices)
  {
    if (reinit_tria)
      this->reinit_tria(boundary_points, place_additional_boundary_vertices);
    MeterBase<dim - 1, spacedim>::internal_reinit();
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
  std::pair<double, Tensor<1, spacedim>>
  SurfaceMeter<dim, spacedim>::compute_flux(
    const int          data_idx,
    const std::string &kernel_name) const
  {
    const auto interpolated_data =
      this->interpolate_vector_field(data_idx, kernel_name);

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

  template class SurfaceMeter<NDIM, NDIM>;
} // namespace fdl
