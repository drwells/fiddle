#ifndef included_fiddle_grid_grid_utilities_h
#define included_fiddle_grid_grid_utilities_h

#include <fiddle/base/config.h>

#include <deal.II/base/point.h>
<<<<<<< HEAD
=======
#include <deal.II/base/quadrature.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
>>>>>>> 07dd614 (add function and tests for intersect line with face/edge)

#include <utility>
#include <vector>

// forward declarations
namespace dealii
{
  template <int, int>
  class Triangulation;
  template <int, int>
  class Mapping;
  template <int>
  class Quadrature;

  namespace parallel
  {
    namespace shared
    {
      template <int, int>
      class Triangulation;
    }
  } // namespace parallel
} // namespace dealii

namespace fdl
{
  using namespace dealii;

  /**
   * Compute the longest edge length of each cell (subject to the provided
   * mapping).
   */
  template <int dim, int spacedim>
  std::vector<float>
  compute_longest_edge_lengths(const Triangulation<dim, spacedim> &tria,
                               const Mapping<dim, spacedim>       &mapping,
                               const Quadrature<1> &line_quadrature);

  /**
   * Collect the edge lengths per element onto each processor.
   */
  template <int dim, int spacedim = dim>
  std::vector<float>
  collect_longest_edge_lengths(
    const parallel::shared::Triangulation<dim, spacedim> &tria,
    const std::vector<float> &local_active_edge_lengths);

  /**
   * Extract a nodeset from an ExodusII file.
   *
   * Returns the node numbers and spatial coordinates.
   *
   * @note If the mesh has duplicated or unused nodes then the node numbers may
   * no longer be meaningful.
   */
  template <int spacedim>
  std::pair<std::vector<unsigned int>, std::vector<Point<spacedim>>>
  extract_nodeset(const std::string &filename, const int nodeset_id);

  /**
   * Compute the centroid of a surface defined by the boundary ids in @p
   * boundary_ids.
   */
  template <int dim>
  Point<dim>
  compute_centroid(const Mapping<dim, dim>               &mapping,
                   const Triangulation<dim, dim>         &tria,
                   const std::vector<types::boundary_id> &boundary_ids);
} // namespace fdl

#endif
