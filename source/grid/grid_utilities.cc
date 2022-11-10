#include <fiddle/base/exceptions.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <vector>

#ifdef DEAL_II_TRILINOS_WITH_SEACAS
#  include <exodusII.h>
#endif

namespace fdl
{
  using namespace dealii;

  namespace internal
  {
    template <int dim, int spacedim>
    std::vector<float>
    compute_longest_edge_lengths(const Triangulation<dim, spacedim> &,
                                 const Mapping<dim, spacedim> &,
                                 const Quadrature<1> &)
    {
      Assert(false, ExcFDLNotImplemented());
      return {};
    }

    template <int spacedim>
    std::vector<float>
    compute_longest_edge_lengths(const Triangulation<2, spacedim> &tria,
                                 const Mapping<2, spacedim>       &mapping,
                                 const Quadrature<1> &line_quadrature)
    {
      std::vector<float> result;
      Assert(tria.get_reference_cells().size() == 1, ExcNotImplemented());
      const ReferenceCell reference_cell = tria.get_reference_cells().front();
      FE_Nothing<2, spacedim> fe_nothing(reference_cell);

      std::vector<float>        line_lengths(tria.n_lines());
      FEFaceValues<2, spacedim> fe_values(mapping,
                                          fe_nothing,
                                          line_quadrature,
                                          update_JxW_values);
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              result.push_back(0.0f);
              // lines are faces in 2D
              for (const auto &line_n : cell->line_indices())
                {
                  float &measure = line_lengths[cell->line_index(line_n)];
                  if (measure == 0.0f)
                    {
                      fe_values.reinit(cell, line_n);
                      for (unsigned int q = 0; q < line_quadrature.size(); ++q)
                        measure += fe_values.JxW(q);
                    }
                  result.back() = std::max(result.back(), measure);
                }
            }
        }

      return result;
    }


    std::vector<float>
    compute_longest_edge_lengths(const Triangulation<3, 3> &tria,
                                 const Mapping<3, 3>       &mapping,
                                 const Quadrature<1> &input_line_quadrature)
    {
      Assert(tria.get_reference_cells().size() == 1, ExcNotImplemented());
      const ReferenceCell reference_cell = tria.get_reference_cells().front();
      // This is quite tricky because deal.II doesn't really support integration
      // along lines in 3D. Work around it by setting up face integrals.
      Assert(reference_cell == ReferenceCells::Tetrahedron ||
               reference_cell == ReferenceCells::Hexahedron,
             ExcNotImplemented());
      const ReferenceCell face_reference_cell =
        reference_cell.face_reference_cell(0);
      // See the notes below explaining why we redefine line_quadrature
      const Quadrature<1> line_quadrature = QGaussLobatto<1>(
        std::max<unsigned int>(2, input_line_quadrature.size()));
      const Quadrature<2> face_quadrature = QProjector<2>::project_to_all_faces(
        reference_cell.face_reference_cell(0), line_quadrature);

      std::vector<float> result;
      FE_Nothing<3, 3>   fe_nothing(reference_cell);
      FEFaceValues<3, 3> fe_values(mapping,
                                   fe_nothing,
                                   face_quadrature,
                                   update_quadrature_points);
      // Cache line values
      std::vector<float> line_lengths(tria.n_lines());
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              result.push_back(0.0f);
              for (const auto &face : cell->face_iterators())
                {
                  // only reinitialize if we actually need to compute something
                  bool needs_reinit = false;
                  for (const auto &line_n : face_reference_cell.line_indices())
                    if (line_lengths[face->line_index(line_n)] == 0.0f)
                      {
                        needs_reinit = true;
                        break;
                      }
                  if (needs_reinit)
                    fe_values.reinit(cell, face);
                  for (const auto &line_n : face_reference_cell.line_indices())
                    {
                      float &measure = line_lengths[face->line_index(line_n)];
                      if (measure == 0.0f)
                        {
                          // This is much less accurate than doing proper
                          // quadrature but is better than any other option I
                          // have right now
                          for (unsigned int q = 0;
                               q < line_quadrature.size() - 1;
                               ++q)
                            {
                              const Point<3> p0 = fe_values.quadrature_point(
                                line_n * line_quadrature.size() + q);
                              const Point<3> p1 = fe_values.quadrature_point(
                                line_n * line_quadrature.size() + q + 1);
                              measure += p0.distance(p1);
                            }
                        }
                      result.back() = std::max(result.back(), measure);
                    }
                }
            }
        }

      return result;
    }
  } // namespace internal



  template <int dim, int spacedim>
  std::vector<float>
  compute_longest_edge_lengths(const Triangulation<dim, spacedim> &tria,
                               const Mapping<dim, spacedim>       &mapping,
                               const Quadrature<1> &line_quadrature)
  {
    return internal::compute_longest_edge_lengths(tria,
                                                  mapping,
                                                  line_quadrature);
  }



  template <int dim, int spacedim = dim>
  std::vector<float>
  collect_longest_edge_lengths(
    const parallel::shared::Triangulation<dim, spacedim> &tria,
    const std::vector<float> &local_active_edge_lengths)
  {
    // TODO: this is very similar to collect_all_active_cell_bboxes - we should
    // generalize the code to gathering of cell data in active cell order
    Assert(
      tria.n_locally_owned_active_cells() == local_active_edge_lengths.size(),
      ExcMessage("There should be an edge length for each local active cell"));

    std::vector<float> global_active_edge_lengths(tria.n_global_active_cells());

    MPI_Comm comm = tria.get_communicator();
    // Exchange number of cells:
    const int        n_procs = Utilities::MPI::n_mpi_processes(comm);
    std::vector<int> lengths_per_proc(n_procs);
    const int        lengths_on_this_proc = tria.n_locally_owned_active_cells();

    int ierr = MPI_Allgather(&lengths_on_this_proc,
                             1,
                             MPI_INT,
                             &lengths_per_proc[0],
                             1,
                             MPI_INT,
                             comm);
    AssertThrowMPI(ierr);
    Assert(std::accumulate(lengths_per_proc.begin(),
                           lengths_per_proc.end(),
                           0u) == tria.n_global_active_cells(),
           ExcMessage("Should be a partition"));

    // Determine indices into temporary array:
    std::vector<int> offsets(n_procs);
    offsets[0] = 0;
    std::partial_sum(lengths_per_proc.begin(),
                     lengths_per_proc.end() - 1,
                     offsets.begin() + 1);
    // Communicate lengths:
    std::vector<float> temp_lengths(tria.n_active_cells());
    ierr = MPI_Allgatherv(local_active_edge_lengths.data(),
                          lengths_on_this_proc,
                          MPI_FLOAT,
                          temp_lengths.data(),
                          lengths_per_proc.data(),
                          offsets.data(),
                          MPI_FLOAT,
                          comm);
    AssertThrowMPI(ierr);

    // Copy to the correct ordering. Keep track of how many cells we have copied
    // from each processor:
    std::vector<int> current_proc_cell_n(n_procs);
    for (const auto &cell : tria.active_cell_iterators())
      {
        const types::subdomain_id this_cell_proc_n =
          tria.get_true_subdomain_ids_of_cells()[cell->active_cell_index()];
        global_active_edge_lengths[cell->active_cell_index()] =
          temp_lengths[offsets[this_cell_proc_n] +
                       current_proc_cell_n[this_cell_proc_n]];
        ++current_proc_cell_n[this_cell_proc_n];
      }

#ifdef DEBUG
    for (const float &length : global_active_edge_lengths)
      Assert(length > 0, ExcMessage("max length should not be zero"));
#endif
    return global_active_edge_lengths;
  }

  template <int spacedim>
  std::pair<std::vector<int>, std::vector<Point<spacedim>>>
  extract_nodeset(const std::string &filename, const int nodeset_id)
  {
#ifdef DEAL_II_TRILINOS_WITH_SEACAS
    // deal.II always uses double precision numbers for geometry
    int component_word_size = sizeof(double);
    // setting to zero uses the stored word size
    int   floating_point_word_size = 0;
    float ex_version               = 0.0;

    const int ex_id = ex_open(filename.c_str(),
                              EX_READ,
                              &component_word_size,
                              &floating_point_word_size,
                              &ex_version);
    AssertThrow(ex_id > 0,
                ExcMessage(
                  "ExodusII failed to open the specified input file."));
    std::vector<char> cell_kind_name(MAX_LINE_LENGTH + 1, '\0');
    int               mesh_dimension   = 0;
    int               n_nodes          = 0;
    int               n_elements       = 0;
    int               n_element_blocks = 0;
    int               n_node_sets      = 0;
    int               n_side_sets      = 0;

    int ierr = ex_get_init(ex_id,
                           cell_kind_name.data(),
                           &mesh_dimension,
                           &n_nodes,
                           &n_elements,
                           &n_element_blocks,
                           &n_node_sets,
                           &n_side_sets);
    AssertThrowExodusII(ierr);
    AssertDimension(mesh_dimension, spacedim);

    int n_nodeset_nodes = 0;
    int n_dist_fact     = 0; // not used
    ierr                = ex_get_set_param(
      ex_id, EX_NODE_SET, nodeset_id, &n_nodeset_nodes, &n_dist_fact);
    AssertThrowExodusII(ierr);

    std::vector<int> node_ids(n_nodeset_nodes);
    ierr = ex_get_set(ex_id, EX_NODE_SET, nodeset_id, node_ids.data(), nullptr);
    AssertThrowExodusII(ierr);

    // Perhaps there is a better way to do this than getting all nodes
    std::vector<double> xs(n_nodes);
    std::vector<double> ys(n_nodes);
    std::vector<double> zs(n_nodes);

    ierr = ex_get_coord(ex_id, xs.data(), ys.data(), zs.data());
    AssertThrowExodusII(ierr);

    std::vector<Point<spacedim>> vertices;
    vertices.reserve(n_nodeset_nodes);
    for (const int &node_id : node_ids)
      {
        auto vertex_n = node_id - 1;
        Assert(vertex_n >= 0 && vertex_n < n_nodes, ExcFDLInternalError());
        switch (spacedim)
          {
            case 1:
              vertices.emplace_back(xs[vertex_n]);
              break;
            case 2:
              vertices.emplace_back(xs[vertex_n], ys[vertex_n]);
              break;
            case 3:
              vertices.emplace_back(xs[vertex_n], ys[vertex_n], zs[vertex_n]);
              break;
            default:
              Assert(spacedim <= 3, ExcNotImplemented());
          }
      }

    ierr = ex_close(ex_id);
    AssertThrowExodusII(ierr);

    return std::make_pair(std::move(node_ids), std::move(vertices));
#else
    (void)filename;
    (void)nodeset_id;
    AssertThrow(false, ExcMessage("Only available with Trilinos + SEACAS"));

    return {};
#endif
  }

  template std::vector<float>
  compute_longest_edge_lengths(const Triangulation<NDIM - 1, NDIM> &,
                               const Mapping<NDIM - 1, NDIM> &,
                               const Quadrature<1> &);
  template std::vector<float>
  compute_longest_edge_lengths(const Triangulation<NDIM, NDIM> &,
                               const Mapping<NDIM, NDIM> &,
                               const Quadrature<1> &);

  template std::vector<float>
  collect_longest_edge_lengths(
    const parallel::shared::Triangulation<NDIM - 1, NDIM> &,
    const std::vector<float> &);

  template std::vector<float>
  collect_longest_edge_lengths(
    const parallel::shared::Triangulation<NDIM, NDIM> &,
    const std::vector<float> &);

  template std::pair<std::vector<int>, std::vector<Point<NDIM>>>
  extract_nodeset<NDIM>(const std::string &filename, const int nodeset_id);
} // namespace fdl
