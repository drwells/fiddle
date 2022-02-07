#include <fiddle/base/exceptions.h>

#include <fiddle/grid/data_in.h>

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#ifdef DEAL_II_TRILINOS_WITH_SEACAS
#  include <exodusII.h>
#endif


#include <boost/algorithm/apply_permutation.hpp>

namespace fdl
{
  using namespace dealii;

#ifdef DEAL_II_TRILINOS_WITH_SEACAS
  namespace
  {
    ReferenceCell
    exodusii_name_to_type(const std::string &type_name,
                          const int          n_nodes_per_element)
    {
      Assert(type_name.size() > 0, ExcInternalError());
      // Try to canonify the name by switching to upper case and removing
      // trailing numbers. This makes, e.g., pyramid, PYRAMID, PYRAMID5, and
      // PYRAMID13 all equal.
      std::string type_name_2 = type_name;
      std::transform(type_name_2.begin(),
                     type_name_2.end(),
                     type_name_2.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      const std::string numbers = "0123456789";
      type_name_2.erase(std::find_first_of(type_name_2.begin(),
                                           type_name_2.end(),
                                           numbers.begin(),
                                           numbers.end()),
                        type_name_2.end());

      if (type_name_2 == "TRI" || type_name_2 == "TRIANGLE")
        return ReferenceCells::Triangle;
      else if (type_name_2 == "QUAD" || type_name_2 == "QUADRILATERAL")
        return ReferenceCells::Quadrilateral;
      else if (type_name_2 == "SHELL")
        {
          if (n_nodes_per_element == 3)
            return ReferenceCells::Triangle;
          else
            return ReferenceCells::Quadrilateral;
        }
      else if (type_name_2 == "TET" || type_name_2 == "TETRA" ||
               type_name_2 == "TETRAHEDRON")
        return ReferenceCells::Tetrahedron;
      else if (type_name_2 == "PYRA" || type_name_2 == "PYRAMID")
        return ReferenceCells::Pyramid;
      else if (type_name_2 == "WEDGE")
        return ReferenceCells::Wedge;
      else if (type_name_2 == "HEX" || type_name_2 == "HEXAHEDRON")
        return ReferenceCells::Hexahedron;

      Assert(false, ExcNotImplemented());
      return ReferenceCells::Invalid;
    }

    template <int spacedim>
    std::vector<Point<spacedim>>
    read_vertices(const int ex_id, const int n_nodes)
    {
      std::vector<double> xs(n_nodes);
      std::vector<double> ys(n_nodes);
      std::vector<double> zs(n_nodes);

      const int ierr = ex_get_coord(ex_id, xs.data(), ys.data(), zs.data());
      AssertThrowExodusII(ierr);

      std::vector<Point<spacedim>> vertices;
      vertices.reserve(n_nodes);
      for (int vertex_n = 0; vertex_n < n_nodes; ++vertex_n)
        {
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

      return vertices;
    }

    int
    get_variable_index(const int            ex_id,
                       const ex_entity_type entity_type,
                       const std::string   &variable_name)
    {
      int n_variables = -1;
      int ierr        = ex_get_variable_param(ex_id, entity_type, &n_variables);
      AssertThrowExodusII(ierr);
      std::vector<char *> names(n_variables);
      const int           max_name_length =
        ex_inquire_int(ex_id, EX_INQ_MAX_READ_NAME_LENGTH);
      for (char *&name : names)
        name = (char *)std::calloc(max_name_length + 1, sizeof(char));
      ierr =
        ex_get_variable_names(ex_id, entity_type, n_variables, names.data());
      AssertThrowExodusII(ierr);

      int variable_index = -1;
      for (int variable_n = 0; variable_n < n_variables; ++variable_n)
        if (names[variable_n] == variable_name)
          {
            // ExodusII indexes from 1
            variable_index = variable_n + 1;
            break;
          }

      AssertThrow(variable_index != -1,
                  ExcMessage("The given variable name " + variable_name +
                             " is not an element variable in the given "
                             "ExodusII file."));

      for (char *&name : names)
        std::free(name);

      return variable_index;
    }

    /**
     * Translate DoF numbers by utilizing the uniqueness of the middle DoF on a
     * given line. Since we know which deal.II vertices the given ExodusII
     * vertices map to, with some extra work we can figure out which deal.II DoF
     * is at the midpoint.
     */
    template <int n_nodes, int n_lines>
    unsigned int
    translate_exodus_line_dof(
      const ReferenceCell              type,
      const std::vector<unsigned int> &vertex_permutation,
      const unsigned int (&exodus_line_dof_to_vertices)[n_nodes][2],
      const unsigned int (&line_n_to_dof_n)[n_lines],
      const unsigned int exodus_line_dof_n)
    {
      // Figure out the deal.II vertices:
      const auto v0 =
        vertex_permutation[exodus_line_dof_to_vertices[exodus_line_dof_n][0]];
      const auto v1 =
        vertex_permutation[exodus_line_dof_to_vertices[exodus_line_dof_n][1]];

      // Figure out the deal.II line:
      unsigned int line_n = numbers::invalid_unsigned_int;
      if (type.get_dimension() == 2)
        {
          auto potential_faces_0 = type.faces_for_given_vertex(v0);
          auto potential_faces_1 = type.faces_for_given_vertex(v1);

          // two vertices can share exactly one face in 2D
          if (potential_faces_0[0] == potential_faces_1[0] ||
              potential_faces_0[0] == potential_faces_1[1])
            line_n = potential_faces_0[0];
          else
            line_n = potential_faces_0[1];
        }
      else
        {
          AssertThrow(false, ExcFDLNotImplemented());
        }

      return line_n_to_dof_n[line_n];
    }

    void
    permute_values(const ReferenceCell        type,
                   const int                  n_nodes_per_element,
                   std::vector<unsigned int> &permutation_to_deal_vertices,
                   std::vector<double>       &component_dof_values)
    {
      AssertDimension(permutation_to_deal_vertices.size(), n_nodes_per_element);
      AssertDimension(component_dof_values.size(), n_nodes_per_element);
      const unsigned int X = numbers::invalid_unsigned_int;
      switch (type)
        {
          // 1D:

          // 2D: map exodus line DoFs to their vertices, translate to deal.II
          // vertices, and then find the corresponding DoF.
          case ReferenceCells::Triangle:
            switch (n_nodes_per_element)
              {
              case 3:
                break; // only vertices, no more to do
              case 6:
                {
                  const unsigned int exodus_line_dof_to_vertices[6][2] = {
                    {X, X}, {X, X}, {X, X}, {0, 1}, {1, 2}, {2, 0}};
                  const unsigned int line_to_dof_n[] = {3, 4, 5};

                  for (int dof_n = type.n_vertices();
                       dof_n < n_nodes_per_element;
                       ++dof_n)
                    {
                      permutation_to_deal_vertices[dof_n] =
                        translate_exodus_line_dof(
                                                  type,
                                                  permutation_to_deal_vertices,
                                                  exodus_line_dof_to_vertices,
                                                  line_to_dof_n,
                                                  dof_n);
                    }
                }
              default:
                AssertThrow(false, ExcFDLNotImplemented());
              }
            break;
          case ReferenceCells::Quadrilateral:
            switch (n_nodes_per_element)
              {
              case 4:
                break; // only vertices, no more to do
              case 9:
                {
                  const unsigned int X = numbers::invalid_unsigned_int;
                  const unsigned int exodus_line_dof_to_vertices[9][2] = {
                    {X, X},
                    {X, X},
                    {X, X},
                    {X, X},
                    {0, 1},
                    {1, 2},
                    {2, 3},
                    {3, 0},
                    {X, X}};

                  const unsigned int line_to_dof_n[] = {4, 5, 6, 7};

                  for (int dof_n = type.n_vertices();
                       dof_n < n_nodes_per_element - 1;
                       ++dof_n)
                    {
                      permutation_to_deal_vertices[dof_n] =
                        translate_exodus_line_dof(
                                                  type,
                                                  permutation_to_deal_vertices,
                                                  exodus_line_dof_to_vertices,
                                                  line_to_dof_n,
                                                  dof_n);
                    }

                  permutation_to_deal_vertices[8] = 8;
                }
                break;
              default:
                AssertThrow(false, ExcFDLNotImplemented());
              }
            break;

          // 3D:
          default:
            AssertThrow(n_nodes_per_element == int(type.n_vertices()),
                        fdl::ExcFDLNotImplemented());
        }

      boost::algorithm::apply_reverse_permutation(
        component_dof_values.begin(),
        component_dof_values.end(),
        permutation_to_deal_vertices.begin(),
        permutation_to_deal_vertices.end());
    }

    template <int dim, int spacedim, typename VectorType>
    void
    read_nodal_components(const int                        ex_id,
                          const std::vector<std::string>  &variable_names,
                          const int                        time_step_n,
                          const DoFHandler<dim, spacedim> &dof_handler,
                          const std::vector<unsigned int> &components,
                          VectorType                      &dof_vector)
    {
      // Read basic mesh information:
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

      const auto       vertices = read_vertices<spacedim>(ex_id, n_nodes);
      std::vector<int> element_block_ids(n_element_blocks);
      ierr = ex_get_ids(ex_id, EX_ELEM_BLOCK, element_block_ids.data());
      AssertThrowExodusII(ierr);


      for (unsigned int component_n = 0; component_n < components.size();
           ++component_n)
        {
          // Permit writing into unghosted vectors by doing a check first
          const IndexSet index_set = dof_vector.locally_owned_elements();

          const int var_index =
            get_variable_index(ex_id, EX_NODAL, variable_names[component_n]);
          // This array could potentially be massive. Try to minimize total
          // memory usage by not loading each component nodal value array at
          // once.
          std::vector<double> nodal_values(n_nodes);
          ierr = ex_get_var(ex_id,
                            time_step_n,
                            EX_NODAL,
                            var_index,
                            1,
                            n_nodes,
                            nodal_values.data());
          AssertThrowExodusII(ierr);

          auto cell = dof_handler.begin_active();
          for (const int element_block_id : element_block_ids)
            {
              std::fill(cell_kind_name.begin(), cell_kind_name.end(), '\0');
              int n_block_elements         = 0;
              int n_nodes_per_element      = 0;
              int n_edges_per_element      = 0;
              int n_faces_per_element      = 0;
              int n_attributes_per_element = 0;

              // Extract element data:
              ierr = ex_get_block(ex_id,
                                  EX_ELEM_BLOCK,
                                  element_block_id,
                                  cell_kind_name.data(),
                                  &n_block_elements,
                                  &n_nodes_per_element,
                                  &n_edges_per_element,
                                  &n_faces_per_element,
                                  &n_attributes_per_element);
              AssertThrowExodusII(ierr);

              const ReferenceCell type =
                exodusii_name_to_type(cell_kind_name.data(),
                                      n_nodes_per_element);
              AssertThrowExodusII(ierr);

              const std::size_t connection_size =
                n_nodes_per_element * n_block_elements;
              // TODO we can support 64-bit indices here - use
              //
              // k = ex_inquire_int(ex_id, EX_INQ_DB_MAX_USED_NAME_LENGTH); and
              // ex_set_max_name_length(ex_id, k);
              std::vector<int> connection(connection_size);
              ierr = ex_get_conn(ex_id,
                                 EX_ELEM_BLOCK,
                                 element_block_id,
                                 connection.data(),
                                 nullptr,
                                 nullptr);
              AssertThrowExodusII(ierr);

              std::vector<types::global_dof_index> cell_dofs;
              std::vector<double>       cell_values(n_nodes_per_element);
              std::vector<unsigned int> exodus_cell_node_ns(
                n_nodes_per_element);
              std::vector<unsigned int> local_exodus_to_deal(
                n_nodes_per_element);
              for (std::size_t node_n = 0; node_n < connection_size;
                   node_n += n_nodes_per_element)
                {
                  if (cell->is_locally_owned())
                    {
                      const FiniteElement<dim, spacedim> &fe = cell->get_fe();
                      cell_dofs.resize(fe.n_dofs_per_cell());
                      cell->get_dof_indices(cell_dofs);
                      std::copy_n(connection.begin() + node_n,
                                  n_nodes_per_element,
                                  exodus_cell_node_ns.begin());
                      for (auto &node_n : exodus_cell_node_ns)
                        node_n -= 1;
                      std::fill(local_exodus_to_deal.begin(),
                                local_exodus_to_deal.end(),
                                numbers::invalid_unsigned_int);
                      for (int i = 0; i < n_nodes_per_element; ++i)
                        cell_values[i] = nodal_values[exodus_cell_node_ns[i]];

                      // At this point (due to renumbering, reorientation, etc)
                      // we are not guaranteed that the node numbers match the
                      // vertex numbers. Hence we reestablish the numbering
                      // based on vertex equality.
                      for (const auto i : type.vertex_indices())
                        for (const auto j : type.vertex_indices())
                          if (vertices[exodus_cell_node_ns[i]] ==
                              cell->vertex(j))
                            local_exodus_to_deal[i] = j;

                      permute_values(type,
                                     n_nodes_per_element,
                                     local_exodus_to_deal,
                                     cell_values);

                      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                        {
                          const auto pair = fe.system_to_component_index(i);
                          if (pair.first == component_n &&
                              index_set.is_element(cell_dofs[i]))
                            dof_vector[cell_dofs[i]] = cell_values[pair.second];
                        }
                    }
                  ++cell;
                }
            }
        }
    }
  } // namespace
#endif

  template <int dim, int spacedim, typename VectorType>
  void
  read_elemental_data(const std::string                  &filename,
                      const Triangulation<dim, spacedim> &tria,
                      const int                           time_step_n,
                      const std::string                  &variable_name,
                      VectorType                         &cell_vector)
  {
    AssertDimension(cell_vector.size(), tria.n_active_cells());
    Assert(tria.n_levels() == 1,
           ExcMessage("This function can only be called on unrefined grids."));
    // According to circa line 2400 of tria.cc, the cells of a Triangulation
    // have the same order of the input file - hence we can just load data
    // into a serial vector and copy that straight over to a parallel deal.II
    // vector.
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

    // Read basic mesh information:
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

#  define CHECK_CENTERS 1
#  ifdef CHECK_CENTERS
    const auto vertices = read_vertices<spacedim>(ex_id, n_nodes);
#  endif

    std::vector<int> element_block_ids(n_element_blocks);
    ierr = ex_get_ids(ex_id, EX_ELEM_BLOCK, element_block_ids.data());
    AssertThrowExodusII(ierr);

    auto deal_cell = tria.begin_active();
    for (const int element_block_id : element_block_ids)
      {
        std::fill(cell_kind_name.begin(), cell_kind_name.end(), '\0');
        int n_block_elements         = 0;
        int n_nodes_per_element      = 0;
        int n_edges_per_element      = 0;
        int n_faces_per_element      = 0;
        int n_attributes_per_element = 0;

        // Extract element data:
        ierr = ex_get_block(ex_id,
                            EX_ELEM_BLOCK,
                            element_block_id,
                            cell_kind_name.data(),
                            &n_block_elements,
                            &n_nodes_per_element,
                            &n_edges_per_element,
                            &n_faces_per_element,
                            &n_attributes_per_element);
        AssertThrowExodusII(ierr);

        const ReferenceCell type =
          exodusii_name_to_type(cell_kind_name.data(), n_nodes_per_element);
        // The number of nodes per element may be larger than what we want to
        // read - for example, if the Exodus file contains a QUAD9 element, we
        // only want to read the first four values and ignore the rest.
        Assert(int(type.n_vertices()) <= n_nodes_per_element,
               ExcInternalError());

        const int var_index =
          get_variable_index(ex_id, EX_ELEM_BLOCK, variable_name);

        // Extract elementwise data:
        std::vector<double> block_element_values(n_block_elements);
        ierr = ex_get_var(ex_id,
                          time_step_n,
                          EX_ELEM_BLOCK,
                          var_index,
                          element_block_id,
                          n_block_elements,
                          block_element_values.data());
        AssertThrowExodusII(ierr);

        {
          const std::size_t connection_size =
            n_nodes_per_element * n_block_elements;
#  if CHECK_CENTERS
          std::vector<int> connection(connection_size);
          ierr = ex_get_conn(ex_id,
                             EX_ELEM_BLOCK,
                             element_block_id,
                             connection.data(),
                             nullptr,
                             nullptr);
          AssertThrowExodusII(ierr);
#  endif

          for (std::size_t node_n = 0; node_n < connection_size;
               node_n += n_nodes_per_element)
            {
              if (deal_cell->is_locally_owned())
                {
#  if CHECK_CENTERS
                  CellData<dim> exodus_cell(type.n_vertices());
                  for (unsigned int i : type.vertex_indices())
                    exodus_cell
                      .vertices[type.exodusii_vertex_to_deal_vertex(i)] =
                      connection[node_n + i] - 1;

                  Point<spacedim> exodus_center;
                  for (const auto index : exodus_cell.vertices)
                    exodus_center += vertices[index];
                  exodus_center /= type.n_vertices();

                  const Point<spacedim> deal_center = deal_cell->center();
                  AssertThrow(
                    (deal_center - exodus_center).norm() < 1e-10,
                    ExcMessage(
                      "The deal.II and ExodusII centers should be the same."));
#  endif
                  cell_vector[deal_cell->active_cell_index()] =
                    block_element_values[node_n / n_nodes_per_element];
                }
              ++deal_cell;
            }
        }
      }

    ierr = ex_close(ex_id);
    AssertThrowExodusII(ierr);

#else
    (void)filename;
    (void)tria;
    (void)time_step_n;
    (void)variable_name;
    (void)cell_vector;
    AssertThrow(false, ExcMessage("Only available with Trilinos + SEACAS"));
#endif
  }



  template <int dim, int spacedim, typename VectorType>
  void
  read_dof_data(const std::string               &filename,
                const DoFHandler<dim, spacedim> &dof_handler,
                const int                        time_step_n,
                const std::vector<std::string>  &variable_names,
                VectorType                      &dof_vector)
  // TODO - make this work with a ComponentMask
  {
#ifdef DEAL_II_TRILINOS_WITH_SEACAS
    Assert(dof_handler.get_triangulation().n_levels() == 1,
           ExcMessage("This function can only be called on unrefined grids."));
    AssertDimension(dof_handler.n_dofs(), dof_vector.size());
    AssertDimension(dof_handler.n_locally_owned_dofs(),
                    dof_vector.locally_owned_size());

    // Partition FEs into elemental and nodal parts. ExodusII does not support
    // variables which have both elemental and nodal parts so we ignore that
    // case. Hence we can examine just the first FE to set up the partitioning
    // on all components.
    std::vector<unsigned int>           elemental_components;
    std::vector<unsigned int>           nodal_components;
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();
    for (unsigned int component = 0; component < fe.n_components(); ++component)
      {
        const FiniteElement<dim, spacedim> &sub_fe =
          fe.get_sub_fe(component, 1);
        if (sub_fe.tensor_degree() == 0)
          elemental_components.push_back(component);
        else
          nodal_components.push_back(component);
      }

    // elemental data:
    for (const unsigned int component : elemental_components)
      {
        Vector<typename VectorType::value_type> component_values(
          dof_handler.get_triangulation().n_active_cells());

        read_elemental_data(filename,
                            dof_handler.get_triangulation(),
                            time_step_n,
                            variable_names[component],
                            component_values);

        std::vector<types::global_dof_index> cell_dofs;
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell_dofs.resize(cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(cell_dofs);
              for (unsigned int i = 0; i < cell_dofs.size(); ++i)
                {
                  const auto pair = fe.system_to_component_index(i);
                  if (pair.first == component)
                    dof_vector[cell_dofs[i]] =
                      component_values[cell->active_cell_index()];
                }
            }
      }

    // nodal data:
    {
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

      std::vector<unsigned int> components(dof_handler.get_fe().n_components());
      std::iota(components.begin(), components.end(), 0u);
      read_nodal_components(ex_id,
                            variable_names,
                            time_step_n,
                            dof_handler,
                            nodal_components,
                            dof_vector);

      const int ierr = ex_close(ex_id);
      AssertThrowExodusII(ierr);
    }
#else
    (void)filename;
    (void)dof_handler;
    (void)time_step_n;
    (void)variable_names;
    (void)dof_vector;
    AssertThrow(false, ExcMessage("Only available with Trilinos + SEACAS"));
#endif
  }

  template void
  read_elemental_data(const std::string                   &filename,
                      const Triangulation<NDIM - 1, NDIM> &tria,
                      const int                            time_step_n,
                      const std::string                   &var_name,
                      Vector<double>                      &cell_vector);

  template void
  read_elemental_data(const std::string               &filename,
                      const Triangulation<NDIM, NDIM> &tria,
                      const int                        time_step_n,
                      const std::string               &var_name,
                      Vector<double>                  &cell_vector);

  template void
  read_elemental_data(const std::string                          &filename,
                      const Triangulation<NDIM - 1, NDIM>        &tria,
                      const int                                   time_step_n,
                      const std::string                          &var_name,
                      LinearAlgebra::distributed::Vector<double> &cell_vector);

  template void
  read_elemental_data(const std::string                          &filename,
                      const Triangulation<NDIM, NDIM>            &tria,
                      const int                                   time_step_n,
                      const std::string                          &var_name,
                      LinearAlgebra::distributed::Vector<double> &cell_vector);

  template void
  read_dof_data(const std::string                &filename,
                const DoFHandler<NDIM - 1, NDIM> &dof_handler,
                const int                         time_step_n,
                const std::vector<std::string>   &var_names,
                Vector<double>                   &dof_vector);

  template void
  read_dof_data(const std::string              &filename,
                const DoFHandler<NDIM, NDIM>   &dof_handler,
                const int                       time_step_n,
                const std::vector<std::string> &var_names,
                Vector<double>                 &dof_vector);

  template void
  read_dof_data(const std::string                          &filename,
                const DoFHandler<NDIM - 1, NDIM>           &dof_handler,
                const int                                   time_step_n,
                const std::vector<std::string>             &var_names,
                LinearAlgebra::distributed::Vector<double> &dof_vector);

  template void
  read_dof_data(const std::string                          &filename,
                const DoFHandler<NDIM, NDIM>               &dof_handler,
                const int                                   time_step_n,
                const std::vector<std::string>             &var_names,
                LinearAlgebra::distributed::Vector<double> &dof_vector);

  template void
  read_dof_data(const std::string                &filename,
                const DoFHandler<NDIM - 1, NDIM> &dof_handler,
                const int                         time_step_n,
                const std::vector<std::string>   &var_names,
                BlockVector<double>              &dof_vector);

  template void
  read_dof_data(const std::string              &filename,
                const DoFHandler<NDIM, NDIM>   &dof_handler,
                const int                       time_step_n,
                const std::vector<std::string> &var_names,
                BlockVector<double>            &dof_vector);

  template void
  read_dof_data(const std::string                               &filename,
                const DoFHandler<NDIM - 1, NDIM>                &dof_handler,
                const int                                        time_step_n,
                const std::vector<std::string>                  &var_names,
                LinearAlgebra::distributed::BlockVector<double> &dof_vector);

  template void
  read_dof_data(const std::string                               &filename,
                const DoFHandler<NDIM, NDIM>                    &dof_handler,
                const int                                        time_step_n,
                const std::vector<std::string>                  &var_names,
                LinearAlgebra::distributed::BlockVector<double> &dof_vector);
} // namespace fdl
