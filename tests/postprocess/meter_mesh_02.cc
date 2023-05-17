#include <fiddle/base/exceptions.h>

#include <fiddle/postprocess/surface_meter.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/SAMRAIDataCache.h>

#include <fstream>

#include "../tests.h"

using namespace dealii;
using namespace SAMRAI;

// Test the meter mesh code for a basic interpolation problem

template <int dim, int spacedim = dim>
void
test(SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer)
{
  auto input_db = app_initializer->getInputDatabase();
  auto test_db  = input_db->getDatabase("test");

  const auto mpi_comm = MPI_COMM_WORLD;
  const auto rank     = Utilities::MPI::this_mpi_process(mpi_comm);

  // setup SAMRAI stuff (its always the same):
  auto tuple           = setup_hierarchy<spacedim>(app_initializer);
  auto patch_hierarchy = std::get<0>(tuple);
  auto f_idx           = std::get<5>(tuple);
  auto g_idx           = std::get<6>(tuple);

  // setup deal.II stuff
  const auto mesh_partitioner =
    parallel::shared::Triangulation<dim, spacedim>::Settings::partition_zorder;
  parallel::shared::Triangulation<dim, spacedim> tria(mpi_comm,
                                                      {},
                                                      false,
                                                      mesh_partitioner);
  GridGenerator::hyper_ball(tria, Point<dim>(), 0.4);
  tria.refine_global(2);
  FunctionParser<spacedim> fp(extract_fp_string(test_db->getDatabase("f")),
                              "PI=" + std::to_string(numbers::PI),
                              "X_0,X_1,X_2");
  FunctionParser<spacedim> gp(extract_fp_string(test_db->getDatabase("g")),
                              "PI=" + std::to_string(numbers::PI),
                              "X_0,X_1,X_2");


  FESystem<dim, spacedim>   fe(FE_Q<dim, spacedim>(1), spacedim);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);
  MappingQ<dim, spacedim> mapping(1);

  // extract part of the Triangulation to serve as the meter mesh
  std::set<unsigned int> bounding_disk_vertex_indices;
  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      for (const unsigned int vertex_n : face->vertex_indices())
        if (std::abs(face->vertex(vertex_n)[spacedim - 1]) < 1e-12)
          bounding_disk_vertex_indices.insert(face->vertex_index(vertex_n));

  // Avoid issues with roundoff in Triangle by perturbing vertices slightly
  GridTools::distort_random(0.1, tria, false);

  std::vector<Point<spacedim>> bounding_disk_points;
  for (const unsigned int vertex_n : bounding_disk_vertex_indices)
    bounding_disk_points.push_back(tria.get_vertices()[vertex_n]);

  // set up position and velocity vectors
  LinearAlgebra::distributed::Vector<double> position(partitioner),
    velocity(partitioner);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           Functions::IdentityFunction<spacedim>(),
                           position);
  for (unsigned int i = 0; i < position.locally_owned_size(); ++i)
    position.local_element(i) += 0.4;
  position.update_ghost_values();

  MappingFEField<dim, spacedim, decltype(position)> position_mapping(
    dof_handler, position);
  VectorTools::interpolate(position_mapping, dof_handler, fp, velocity);
  velocity.update_ghost_values();

  fdl::SurfaceMeter<dim, spacedim> meter_mesh(mapping,
                                              dof_handler,
                                              bounding_disk_points,
                                              patch_hierarchy,
                                              position,
                                              velocity);
  AssertThrow(meter_mesh.uses_codim_zero_mesh(), fdl::ExcFDLInternalError());

  std::ofstream output;
  if (rank == 0)
    {
      output.open("output");
    }

  // write SAMRAI data:
  {
    app_initializer->getVisItDataWriter()->writePlotData(patch_hierarchy,
                                                         0,
                                                         0.0);
  }

  // plot native solution:
  {
    const Triangulation<dim - 1, spacedim> &meter_tria =
      meter_mesh.get_triangulation();
    const auto interpolated_F =
      meter_mesh.interpolate_vector_field(f_idx, "BSPLINE_3");
    const auto interpolated_G =
      meter_mesh.interpolate_scalar_field(g_idx, "BSPLINE_3");
    DataOut<dim - 1, spacedim> data_out;
    data_out.add_data_vector(meter_mesh.get_vector_dof_handler(),
                             interpolated_F,
                             "F");
    data_out.add_data_vector(meter_mesh.get_scalar_dof_handler(),
                             interpolated_G,
                             "G");

    double global_f_error = 0.0;
    {
      Vector<double> error(meter_tria.n_active_cells());
      interpolated_F.update_ghost_values();
      VectorTools::integrate_difference(meter_mesh.get_mapping(),
                                        meter_mesh.get_vector_dof_handler(),
                                        interpolated_F,
                                        fp,
                                        error,
                                        QGauss<dim - 1>(3),
                                        VectorTools::L2_norm);
      global_f_error = VectorTools::compute_global_error(meter_tria,
                                                         error,
                                                         VectorTools::L2_norm);
    }
    double global_g_error = 0.0;
    {
      Vector<double> error(meter_tria.n_active_cells());
      interpolated_G.update_ghost_values();
      VectorTools::integrate_difference(meter_mesh.get_mapping(),
                                        meter_mesh.get_scalar_dof_handler(),
                                        interpolated_G,
                                        gp,
                                        error,
                                        QGauss<dim - 1>(3),
                                        VectorTools::L2_norm);
      global_g_error = VectorTools::compute_global_error(meter_tria,
                                                         error,
                                                         VectorTools::L2_norm);
    }

    Tensor<1, spacedim> mean_velocity;
    double              flux = 0.0;
    double              area = 0.0;
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
        const Triangulation<dim - 1, spacedim> &meter_tria =
          meter_mesh.get_triangulation();
        Quadrature<dim - 2>           face_quadrature(points, weights);
        QGaussSimplex<dim - 1>        cell_quadrature(3);
        FE_Nothing<dim - 1, spacedim> fe_nothing(
          meter_tria.get_reference_cells()[0]);
        FEFaceValues<dim - 1, spacedim> face_values(meter_mesh.get_mapping(),
                                                    fe_nothing,
                                                    face_quadrature,
                                                    update_JxW_values |
                                                      update_quadrature_points);
        FEValues<dim - 1, spacedim>     fe_values(meter_mesh.get_mapping(),
                                              fe_nothing,
                                              cell_quadrature,
                                              update_normal_vectors |
                                                update_quadrature_points |
                                                update_JxW_values);
        for (const auto &cell : meter_tria.active_cell_iterators())
          {
            fe_values.reinit(cell);
            for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
              {
                Tensor<1, spacedim> fp_value;
                for (unsigned int d = 0; d < dim; ++d)
                  fp_value[d] = fp.value(fe_values.quadrature_point(q), d);
                flux +=
                  fp_value * fe_values.normal_vector(q) * fe_values.JxW(q);
              }

            for (unsigned int face_no : cell->face_indices())
              if (cell->face(face_no)->at_boundary())
                {
                  face_values.reinit(cell, face_no);
                  for (unsigned int d = 0; d < spacedim; ++d)
                    {
                      mean_velocity[d] +=
                        fp.value(face_values.get_quadrature_points()[0], d) *
                        face_values.get_JxW_values()[0];
                      mean_velocity[d] +=
                        fp.value(face_values.get_quadrature_points()[1], d) *
                        face_values.get_JxW_values()[1];
                    }
                  area += face_values.get_JxW_values()[0];
                  area += face_values.get_JxW_values()[1];
                }
          }
        mean_velocity /= area;
      }

    const double meter_flux = meter_mesh.compute_flux(f_idx, "BSPLINE_3");
    const double meter_centroid_g =
      meter_mesh.compute_centroid_value(g_idx, "BSPLINE_3");
    if (rank == 0)
      output << "number of hull points = " << bounding_disk_points.size()
             << std::endl
             << "number of vertices = " << meter_tria.get_vertices().size()
             << std::endl
             << "number of active cells = " << meter_tria.n_active_cells()
             << std::endl
             << "centroid = " << meter_mesh.get_centroid() << std::endl
             << "computed centroid G = " << gp.value(meter_mesh.get_centroid())
             << std::endl
             << "   meter centroid G = " << meter_centroid_g << std::endl
             << "global error in F = " << global_f_error << std::endl
             << "global error in G = " << global_g_error << std::endl
             << "computed mean velocity = " << mean_velocity << std::endl
             << "   meter mean velocity = " << meter_mesh.get_mean_velocity()
             << std::endl
             << "computed flux = " << flux << std::endl
             << "   meter flux = " << meter_flux << std::endl
             << std::endl
             << "global error in mean velocity = "
             << (mean_velocity - meter_mesh.get_mean_velocity()).norm()
             << std::endl;

    Vector<float> subdomain(meter_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = meter_tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "meter-tria", 0, mpi_comm, 2, 8);
  }
}

int
main(int argc, char **argv)
{
  IBTK::IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
  SAMRAI::tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "multilevel_fe_01.log");

  test<3>(app_initializer);
}
