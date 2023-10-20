#include <fiddle/base/exceptions.h>

#include <fiddle/grid/surface_tria.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/householder.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_creator.h>

#include <fstream>
#include <memory>
#include <vector>

#include "triangle.h"

namespace fdl
{
  using namespace dealii;

  void
  triangulate_segments(const std::vector<Point<2>>   &boundary_vertices,
                       Triangulation<2>              &tria,
                       const Triangle::AdditionalData additional_data)
  {
    AssertThrow(boundary_vertices.size() >= 3, ExcFDLNotImplemented());

    // C structures are not initialized by default - zero everything
    triangulateio in, out;
    std::memset(&in, 0, sizeof(in));
    std::memset(&out, 0, sizeof(out));

    in.numberofpoints          = boundary_vertices.size();
    in.numberofpointattributes = 0;

    std::vector<double> pointlist;
    pointlist.reserve(boundary_vertices.size() * 2);
    for (const Point<2> &p : boundary_vertices)
      {
        if (additional_data.regularize_input)
          {
            pointlist.push_back(float(p[0]));
            pointlist.push_back(float(p[1]));
          }
        else
          {
            pointlist.push_back(p[0]);
            pointlist.push_back(p[1]);
          }
      }
    in.pointlist = pointlist.data();

    std::vector<int> segments(boundary_vertices.size() * 2,
                              std::numeric_limits<int>::max());
    in.numberofsegments = boundary_vertices.size();
    in.segmentlist      = segments.data();

    // Determine segments based on the closest vertex.
    {
      std::vector<int> n_vertex_segments(boundary_vertices.size());
      int              previous_vertex_no = std::numeric_limits<int>::max();
      int              vertex_no          = 0;
      for (int segment_n = 0; segment_n < in.numberofsegments; ++segment_n)
        {
          int    other_vertex_no = std::numeric_limits<int>::max();
          double distance        = std::numeric_limits<double>::max();

          // first try to find a point which is not part of any segment and,
          // if we are on at least the second point, is 'ahead' of the current
          // point (the angle between this segment and the last is less than
          // 180 degrees)
          for (int v = 0; v < in.numberofpoints; ++v)
            // each vertex can only be part of at most two segments. Also make
            // sure we didn't already pick this segment
            if (v != vertex_no && n_vertex_segments[v] == 0 &&
                segments[v * 2 + 1] != vertex_no)
              {
                Tensor<1, 2> t0;
                if (previous_vertex_no == std::numeric_limits<int>::max())
                  // If there is no previous point, arbitrarily assume we are
                  // going from vertex 1 (whereever that may be)
                  t0 = boundary_vertices[vertex_no] - boundary_vertices[1];
                else
                  t0 = boundary_vertices[vertex_no] -
                       boundary_vertices[previous_vertex_no];
                const Tensor<1, 2> t1 =
                  boundary_vertices[v] - boundary_vertices[vertex_no];
                const bool   in_cone = t0 * t1 > 0;
                const double new_distance =
                  boundary_vertices[vertex_no].distance_square(
                    boundary_vertices[v]);
                if (in_cone && new_distance < distance)
                  {
                    distance        = new_distance;
                    other_vertex_no = v;
                  }
              }
          // if that failed, try again and permit points outside the cone:
          if (other_vertex_no == std::numeric_limits<int>::max())
            for (int v = 0; v < in.numberofpoints; ++v)
              // each vertex can only be part of at most two segments. Also make
              // sure we didn't already pick this segment
              if (v != vertex_no && n_vertex_segments[v] == 0 &&
                  segments[v * 2 + 1] != vertex_no)
                {
                  const double new_distance =
                    boundary_vertices[vertex_no].distance_square(
                      boundary_vertices[v]);
                  if (new_distance < distance)
                    {
                      distance        = new_distance;
                      other_vertex_no = v;
                    }
                }

          // if that failed, try again and permit points which are already
          // part of one segment:
          if (other_vertex_no == std::numeric_limits<int>::max())
            for (int v = 0; v < in.numberofpoints; ++v)
              // each vertex can only be part of at most two segments. Also
              // make sure we didn't already pick this segment
              if (v != vertex_no && n_vertex_segments[v] == 1 &&
                  segments[v * 2 + 1] != vertex_no)
                {
                  const double new_distance =
                    boundary_vertices[vertex_no].distance_square(
                      boundary_vertices[v]);
                  if (new_distance < distance)
                    {
                      distance        = new_distance;
                      other_vertex_no = v;
                    }
                }

          AssertThrow(other_vertex_no != std::numeric_limits<int>::max(),
                      ExcMessage("unable to find other vertex for segment"));
          segments[vertex_no * 2]     = vertex_no;
          segments[vertex_no * 2 + 1] = other_vertex_no;
          n_vertex_segments[vertex_no] += 1;
          n_vertex_segments[other_vertex_no] += 1;

          // go to the next loop iteration
          previous_vertex_no = vertex_no;
          vertex_no          = other_vertex_no;
        }

      for (int vertex_no = 0; vertex_no < in.numberofpoints; ++vertex_no)
        AssertThrow(n_vertex_segments[vertex_no] == 2,
                    ExcMessage(
                      "Each vertex should be part of exactly two segments."));
    }

    // use a (p)SLG, index from (z)ero, (Q)uiet output, do a (C)onsistency
    // check, element (q)uality (min angle in degrees)
    std::string flags("pzQCq");
    Assert(additional_data.min_angle > 0.0,
           ExcMessage("The minimum angle must be larger than zero.")) flags +=
      std::to_string(additional_data.min_angle);
    if (additional_data.target_element_area ==
        std::numeric_limits<double>::max())
      {
        const double dx = boundary_vertices[segments[0]].distance(
          boundary_vertices[segments[1]]);
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

    if (additional_data.regularize_input)
      {
#ifdef DEBUG
        for (int i = 0; i < in.numberofpoints; ++i)
          {
            Assert(
              (out.pointlist[2 * i] == float(boundary_vertices[i][0])) &&
                (out.pointlist[2 * i + 1] == float(boundary_vertices[i][1])),
              ExcMessage(
                "The truncated vertices should match the output vertices."));
          }
#endif
        vertices = boundary_vertices;
        for (int i = in.numberofpoints; i < out.numberofpoints; ++i)
          vertices.emplace_back(out.pointlist[2 * i], out.pointlist[2 * i + 1]);
      }
    else
      {
        for (int i = 0; i < out.numberofpoints; ++i)
          vertices.emplace_back(out.pointlist[2 * i], out.pointlist[2 * i + 1]);
      }
    AssertDimension(vertices.size(), out.numberofpoints);

    GridTools::invert_cells_with_negative_measure(vertices, cell_data);
    if (additional_data.apply_fixup_routines)
      {
        std::vector<unsigned int> all_vertices;
        GridTools::delete_unused_vertices(vertices, cell_data, sub_cell_data);
        // This should not be needed (Triangle won't duplicate vertices) but
        // lets do it anyway
        GridTools::delete_duplicated_vertices(vertices,
                                              cell_data,
                                              sub_cell_data,
                                              all_vertices);
      }
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
  create_planar_triangulation(const std::vector<Point<3>>   &points,
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
    std::vector<Point<2>> boundary_points2;
    for (const Point<3> &p : points)
      boundary_points2.emplace_back((p - points[0]) * v0, (p - points[0]) * v1);

    // 3. set up the triangulation in the z = 0 plane
    triangulate_segments(boundary_points2, tria2, additional_data);
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


  template <int dim, int spacedim>
  void
  fit_boundary_vertices(const std::vector<Point<spacedim>> &new_vertices,
                        Triangulation<dim, spacedim>       &tria)
  {
    Assert(tria.get_communicator() == MPI_COMM_SELF, ExcNotImplemented());
    Assert(tria.get_reference_cells().size() == 1, ExcNotImplemented());
    Assert(tria.n_levels() == 1, ExcNotImplemented());

#ifdef DEBUG
    for (const auto &face : tria.active_face_iterators())
      if (face->at_boundary())
        for (const auto v : face->vertex_indices())
          AssertIndexRange(face->vertex_index(v), new_vertices.size());
#endif

    const auto         reference_cell = tria.get_reference_cells()[0];
    const unsigned int degree         = 1;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe;
    if (reference_cell.is_hyper_cube())
      fe = std::make_unique<FE_Q<dim, spacedim>>(degree);
    else if (reference_cell.is_simplex())
      fe = std::make_unique<FE_SimplexP<dim, spacedim>>(degree);
    else
      Assert(false, ExcNotImplemented());

    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(*fe);
    const Quadrature<dim> quadrature =
      reference_cell.template get_gauss_type_quadrature<dim>(degree + 1);
    const Mapping<dim, spacedim> &mapping =
      reference_cell.template get_default_linear_mapping<dim, spacedim>();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityPattern sp;
    sp.copy_from(dsp);

    SparseMatrix<double> laplace_matrix(sp);
    MatrixCreator::create_laplace_matrix(mapping,
                                         dof_handler,
                                         quadrature,
                                         laplace_matrix);

    // compute new vertex locations one component at a time
    std::vector<Point<spacedim>> new_vertex_positions(
      tria.get_vertices().size());
    for (unsigned int d = 0; d < spacedim; ++d)
      {
        // Set up constraints:
        AffineConstraints<double> constraints;
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->at_boundary())
            {
              for (unsigned int face_no : cell->face_indices())
                if (cell->face(face_no)->at_boundary())
                  for (const auto face_vertex_no :
                       cell->face(face_no)->vertex_indices())
                    {
                      const auto vertex_no =
                        reference_cell.face_to_cell_vertices(
                          face_no,
                          face_vertex_no,
                          cell->combined_face_orientation(face_no));

                      const auto vertex_dof =
                        cell->vertex_dof_index(vertex_no, 0);
                      constraints.add_line(vertex_dof);
                      constraints.set_inhomogeneity(
                        vertex_dof,
                        new_vertices[cell->vertex_index(vertex_no)][d]);
                    }
            }
        constraints.close();

        SparseMatrix<double> constrained_laplace_matrix(sp);
        Vector<double>       constrained_rhs(dof_handler.n_dofs());

        constrained_laplace_matrix.copy_from(laplace_matrix);
        constraints.condense(constrained_laplace_matrix, constrained_rhs);
        SolverControl      solver_control(1000, 1e-12);
        SolverCG<>         solver(solver_control);
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(constrained_laplace_matrix, 1.2);
        Vector<double> solution(dof_handler.n_dofs());
        solver.solve(constrained_laplace_matrix,
                     solution,
                     constrained_rhs,
                     preconditioner);
        constraints.distribute(solution);


        for (const auto &cell : dof_handler.active_cell_iterators())
          for (unsigned int vertex_no : cell->vertex_indices())
            new_vertex_positions[cell->vertex_index(vertex_no)][d] =
              solution[cell->vertex_dof_index(vertex_no, 0)];
      }

    for (auto &cell : tria.active_cell_iterators())
      for (unsigned int vertex_no : cell->vertex_indices())
        cell->vertex(vertex_no) =
          new_vertex_positions[cell->vertex_index(vertex_no)];
    tria.signals.mesh_movement();
  }

  // instantiate all of them: why not?

  template void
  fit_boundary_vertices(const std::vector<Point<2>> &, Triangulation<1, 2> &);

  template void
  fit_boundary_vertices(const std::vector<Point<2>> &, Triangulation<2, 2> &);

  template void
  fit_boundary_vertices(const std::vector<Point<3>> &, Triangulation<2, 3> &);

  template void
  fit_boundary_vertices(const std::vector<Point<3>> &, Triangulation<3, 3> &);
} // namespace fdl
