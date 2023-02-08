#include <fiddle/grid/surface_tria.h>

#include <deal.II/base/tensor.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/householder.h>

#include "triangle.h"

namespace fdl
{
  using namespace dealii;

  void
  triangulate_convex(const std::vector<Point<2>>   &hull_vertices,
                     Triangulation<2>              &tria,
                     const Triangle::AdditionalData additional_data)
  {
    static_assert(sizeof(Point<2>) == 2 * sizeof(double),
                  "format should be packed");

    // C structures are not initialized by default - zero everything
    triangulateio in, out;
    std::memset(&in, 0, sizeof(in));
    std::memset(&out, 0, sizeof(out));

    in.numberofpoints          = hull_vertices.size();
    in.numberofpointattributes = 0;
    in.pointlist               = const_cast<double *>(
      reinterpret_cast<const double *>(hull_vertices.data()));
    // follow from libMesh - number of segments = number of holes for convex
    // hulls
    in.numberofsegments = 0;

    // (c)onvex hull, index from (z)ero, (Q)uiet output, element (q)uality (min
    // angle, degrees)
    std::string flags("czQq");
    Assert(additional_data.min_angle > 0.0,
           ExcMessage("The minimum angle must be larger than zero.")) flags +=
      std::to_string(additional_data.min_angle);
    if (additional_data.target_element_area ==
        std::numeric_limits<double>::max())
      {
        const double dx = hull_vertices[0].distance(hull_vertices[1]);
        // target element (a)rea
        flags += "a" + std::to_string(std::sqrt(3.0) / 4.0 * dx * dx);
      }
    else
      flags += "a" + std::to_string(additional_data.target_element_area);
    if (additional_data.place_additional_boundary_vertices == false)
      flags += "Y";

    triangulate(const_cast<char *>(flags.c_str()), &in, &out, nullptr);

    std::vector<CellData<2>> cell_data;
    std::vector<Point<2>>    vertices;
    SubCellData              sub_cell_data;

    for (int i = 0; i < out.numberoftriangles; ++i)
      {
        cell_data.emplace_back();
        cell_data.back().vertices.resize(3);
        cell_data.back().vertices[0] = out.trianglelist[3 * i];
        cell_data.back().vertices[1] = out.trianglelist[3 * i + 1];
        cell_data.back().vertices[2] = out.trianglelist[3 * i + 2];
      }

    for (int i = 0; i < out.numberofpoints; ++i)
      vertices.emplace_back(out.pointlist[2 * i], out.pointlist[2 * i + 1]);

    // The input list of vertices may contain duplicates - Triangle knows how
    // to handle that (it doesn't use them), but make sure we delete them
    // ourselves before continuing
    std::vector<unsigned int> all_vertices;
    GridTools::delete_unused_vertices(vertices,
                                      cell_data,
                                      sub_cell_data);
    // This should not be needed (Triangle won't duplicate vertices) but lets
    // do it anyway
    GridTools::delete_duplicated_vertices(vertices,
                                          cell_data,
                                          sub_cell_data,
                                          all_vertices);
    tria.create_triangulation(vertices, cell_data, sub_cell_data);

    // Free everything triangle may have allocated
    trifree(out.pointlist);
    trifree(out.pointattributelist);
    trifree(out.pointmarkerlist);
    trifree(out.trianglelist);
    trifree(out.triangleattributelist);
    trifree(out.trianglearealist);
    trifree(out.neighborlist);
    trifree(out.segmentlist);
    trifree(out.segmentmarkerlist);
    trifree(out.holelist);
    trifree(out.regionlist);
    trifree(out.edgelist);
    trifree(out.edgemarkerlist);
    trifree(out.normlist);
  }

  Tensor<1, 3>
  setup_planar_meter_mesh(const std::vector<Point<3>>   &points,
                          Triangulation<2, 3>           &tria,
                          const Triangle::AdditionalData additional_data)
  {
    // 1. Do a least-squares fit of the points to a plane: z = a x + b y + c
    FullMatrix<double> data(points.size(), 3);
    Vector<double>     rhs(points.size());
    for (std::size_t point_n = 0; point_n < points.size(); ++point_n)
      {
        for (unsigned int d = 0; d < 2; ++d)
          data(point_n, d) = points[point_n][d];
        data(point_n, 2) = 1.0;
        rhs[point_n]     = points[point_n][2];
      }
    Householder<double> householder(data);
    Vector<double>      coeffs(3);
    householder.least_squares(coeffs, rhs);

    // Get two orthogonal planar vectors.
    Tensor<1, 3> normal{{-coeffs[0], -coeffs[1], 1.0}};
    normal /= normal.norm();
    Tensor<1, 3> v0{{1.0, 0.0, coeffs[0]}};
    v0 /= v0.norm();

    Assert(std::abs(v0 * normal) < 1e-12, ExcInternalError());
    Tensor<1, 3> v1 = cross_product_3d(normal, v0);
    v1 /= v1.norm();

    // 2. Express the new points in plane coordinates. All of these
    // transformations are linear so shift things so that the first point is the
    // origin.
    Triangulation<2>      tria2;
    std::vector<Point<2>> hull_points2;
    for (const Point<3> &p : points)
      hull_points2.emplace_back((p - points[0]) * v0, (p - points[0]) * v1);

    // 3. set up the triangulation in the z = 0 plane
    triangulate_convex(hull_points2, tria2, additional_data);
    std::vector<Point<2>>    points2;
    std::vector<CellData<2>> cells2;
    SubCellData              subcell_data2;
    std::tie(points2, cells2, subcell_data2) =
      GridTools::get_coarse_mesh_description(tria2);

    // 4. project points back into the plane
    std::vector<Point<3>> points3;
    for (const Point<2> &p : points2)
      points3.emplace_back(points[0] + p[0] * v0 + p[1] * v1);

    // Set up the meter mesh
    tria.create_triangulation(points3, cells2, subcell_data2);
    return normal;
  }
} // namespace fdl
