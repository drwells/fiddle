#include <fiddle/grid/surface_tria.h>

#include <deal.II/grid/grid_out.h>

#include <fstream>

// Test out our Triangle wrapper for generating surface meshes.

int
main()
{
  using namespace dealii;

  std::ofstream output("output");

  output << "Planar example\n";
  {
    Triangulation<2>      tria;
    std::vector<Point<2>> vertices;
    vertices.emplace_back(1.0, 0.0);
    vertices.emplace_back(0.8, 0.25);
    vertices.emplace_back(0.6, 0.6);
    vertices.emplace_back(0.25, 0.7);
    vertices.emplace_back(0.0, 1.0);
    vertices.emplace_back(0.0, 0.0);

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area                = 0.0125;
    additional_data.place_additional_boundary_vertices = true;
    fdl::triangulate_segments(vertices, tria, additional_data);

    GridOut().write_vtk(tria, output);
  }

  output << "3D example\n";
  {
    Triangulation<2, 3>   tria;
    std::vector<Point<3>> vertices;
    vertices.emplace_back(1.0, 0.0, 1.0);
    vertices.emplace_back(0.8, 0.25, 0.8);
    vertices.emplace_back(0.6, 0.6, 0.6);
    vertices.emplace_back(0.25, 0.7, 0.25);
    vertices.emplace_back(0.0, 1.0, 0.0);
    vertices.emplace_back(0.0, 0.0, 0.0);

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area                = 0.0125;
    additional_data.place_additional_boundary_vertices = true;
    fdl::create_planar_triangulation(vertices, tria, additional_data);

    GridOut().write_vtk(tria, output);
  }

  output << "3D example, small elements\n";
  {
    Triangulation<2, 3>   tria;
    std::vector<Point<3>> vertices;
    const unsigned int    n_points = 10;
    for (unsigned int i = 0; i < n_points; ++i)
      vertices.emplace_back(std::cos(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(2.0 * numbers::PI * i / double(n_points)),
                            1.0);

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area =
      (vertices[1] - vertices[0]).norm() / 8.0;
    additional_data.place_additional_boundary_vertices = true;
    fdl::create_planar_triangulation(vertices, tria, additional_data);

    GridOut().write_vtk(tria, output);
  }

  output << "3D example, no extra boundary nodes\n";
  {
    Triangulation<2, 3>   tria;
    std::vector<Point<3>> vertices;
    const unsigned int    n_points = 10;
    for (unsigned int i = 0; i < n_points; ++i)
      vertices.emplace_back(std::cos(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(2.0 * numbers::PI * i / double(n_points)),
                            1.0);

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area =
      (vertices[1] - vertices[0]).norm() / 8.0;
    additional_data.place_additional_boundary_vertices = false;
    fdl::create_planar_triangulation(vertices, tria, additional_data);

    GridOut().write_vtk(tria, output);
  }

  output << "3D example, slanted\n";
  {
    Triangulation<2, 3>   tria;
    std::vector<Point<3>> vertices;
    const unsigned int    n_points = 10;
    for (unsigned int i = 0; i < n_points; ++i)
      vertices.emplace_back(std::cos(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(2.0 * numbers::PI * i / double(n_points)));

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area =
      (vertices[1] - vertices[0]).norm() / 4.0;
    additional_data.place_additional_boundary_vertices = false;
    fdl::create_planar_triangulation(vertices, tria, additional_data);

    // this shouldn't do anything since the mesh is already planar
    fdl::fit_boundary_vertices(vertices, tria);

    GridOut().write_vtk(tria, output);
  }

  output << "3D example, slanted 2\n";
  {
    Triangulation<2, 3>   tria;
    std::vector<Point<3>> vertices;
    const unsigned int    n_points = 20;
    for (unsigned int i = 0; i < n_points; ++i)
      vertices.emplace_back(std::cos(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(2.0 * numbers::PI * i / double(n_points)),
                            std::sin(4.0 * numbers::PI * i / double(n_points)));

    fdl::Triangle::AdditionalData additional_data;
    additional_data.target_element_area =
      (vertices[1] - vertices[0]).norm() / 4.0;
    additional_data.place_additional_boundary_vertices = false;
    fdl::create_planar_triangulation(vertices, tria, additional_data);

    fdl::fit_boundary_vertices(vertices, tria);

    GridOut().write_vtk(tria, output);
  }
}
