#ifndef included_fiddle_grid_grid_utilities_h
#define included_fiddle_grid_grid_utilities_h

#include <fiddle/base/config.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <utility>
#include <vector>

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
   */
  template <int spacedim>
  std::pair<std::vector<int>, std::vector<Point<spacedim>>>
  extract_nodeset(const std::string &filename,
                  const int          nodeset_id);
} // namespace fdl

#endif
