#ifndef included_fiddle_grid_data_in_h
#define included_fiddle_grid_data_in_h

#include <fiddle/base/config.h>

#include <string>

// forward declarations
namespace dealii
{
  template <int, int>
  class Triangulation;
  template <int, int>
  class DoFHandler;
} // namespace dealii

namespace fdl
{
  using namespace dealii;

  /**
   * @brief Read elemental data from an ExodusII file.
   *
   * This function loads data stored in the ExodusII file @p filename at
   * timestep @p time_step and variable @p var_index corresponding to the active
   * cells of the Triangulation provided as an argument. @p cell_vector can be
   * ghosted or unghosted.
   *
   * This function is only available if deal.II is configured with Trilinos
   * with SEACAS.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  read_elemental_data(const std::string                  &filename,
                      const Triangulation<dim, spacedim> &tria,
                      const int                           time_step_n,
                      const std::string                  &var_name,
                      VectorType                         &cell_vector);

  /**
   * @brief Read DoF data from an ExodusII file. At this time higher-order 3D
   * elements are not yet implemented.
   *
   * This function does not support ComponentMask yet - you must load one
   * variable for each component of the DoFHandler.
   *
   * This function loads data stored in the ExodusII file @p filename at
   * timestep @p time_step and variable @p var_index corresponding to the
   * DoFHandler provided as an argument. @p cell_vector can be ghosted or
   * unghosted.
   *
   * This function is only available if deal.II is configured with Trilinos
   * with SEACAS.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  read_dof_data(const std::string               &filename,
                const DoFHandler<dim, spacedim> &dof_handler,
                const int                        time_step_n,
                const std::vector<std::string>  &variable_names,
                VectorType                      &dof_vector);
} // namespace fdl

#endif
