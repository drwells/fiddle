#include <fiddle/base/config.h>

#include <fiddle/grid/overlap_tria.h>

#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

namespace fdl
{
  using namespace dealii;

  /**
   * Compute the dof translation between degrees of freedom assigned to the
   * OverlapTriangulation and the equivalent degrees of freedom assigned to
   * the native triangulation.
   *
   * Since the OverlapTriangulation is an inherently local object, the dof
   * indices on its DoFHandler form a contiguous index space starting at zero.
   * Hence we can compute the equivalent global dofs as an array, where the
   * overlap dofs are the array indices and the native dofs are the values.
   *
   * @note This function is collective over the communicator used by @p
   * native_dof_handler.
   */
  template <int dim, int spacedim = dim>
  std::vector<types::global_dof_index>
  compute_overlap_to_native_dof_translation(
    const fdl::OverlapTriangulation<dim, spacedim> &overlap_tria,
    const DoFHandler<dim, spacedim> &               overlap_dof_handler,
    const DoFHandler<dim, spacedim> &               native_dof_handler);
} // namespace fdl
