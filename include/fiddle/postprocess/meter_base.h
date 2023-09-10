#ifndef included_fiddle_postprocess_meter_base_h
#define included_fiddle_postprocess_meter_base_h

#include <fiddle/base/config.h>

#include <fiddle/interaction/nodal_interaction.h>

#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <tbox/Pointer.h>

#include <memory>
#include <utility>
#include <vector>

namespace SAMRAI
{
  namespace hier
  {
    template <int>
    class PatchHierarchy;
  }
} // namespace SAMRAI


namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  /**
   * Base class for the meter classes (SurfaceMeter and VolumeMeter).
   */
  template <int dim, int spacedim = dim>
  class MeterBase
  {
  public:
    /**
     * Constructor.
     */
    MeterBase(tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy);

    /**
     * Constructor. Copies an existing Triangulation.
     */
    MeterBase(const Triangulation<dim, spacedim>           &tria,
              tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy);

    /* @name object access
     * @{
     */

    /**
     * Return a reference to the meter Triangulation. This triangulation is
     * not in reference coordinates: instead its absolute position is
     * determined by the position vector specified to the constructor or
     * reinit().
     */
    const Triangulation<dim, spacedim> &
    get_triangulation() const;

    /**
     * Return a reference to the Mapping used on the meter mesh.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const;

    /**
     * Return a reference to the DoFHandler for scalar fields.
     */
    const DoFHandler<dim, spacedim> &
    get_scalar_dof_handler() const;

    /**
     * Return a reference to the DoFHandler for vector fields.
     */
    const DoFHandler<dim, spacedim> &
    get_vector_dof_handler() const;

    /**
     * Return the centroid of the meter mesh. This point may not be inside the
     * mesh.
     */
    virtual Point<spacedim>
    get_centroid() const;

    /** @} */

    /* @name FSI
     * @{
     */

    /**
     * Interpolate the value of some scalar field at the centroid.
     */
    virtual double
    compute_centroid_value(const int          data_idx,
                           const std::string &kernel_name) const;

    /**
     * Compute the mean value of some scalar-valued quantity.
     *
     * @param[in] data_idx Some data index corresponding to data on the
     * Cartesian grid. This class will copy the data into a scratch index and
     * update ghost data.
     */
    virtual double
    compute_mean_value(const int          data_idx,
                       const std::string &kernel_name) const;

    /**
     * Interpolate a scalar-valued quantity.
     */
    virtual LinearAlgebra::distributed::Vector<double>
    interpolate_scalar_field(const int          data_idx,
                             const std::string &kernel_name) const;

    /**
     * Interpolate a vector-valued quantity.
     */
    virtual LinearAlgebra::distributed::Vector<double>
    interpolate_vector_field(const int          data_idx,
                             const std::string &kernel_name) const;

    /**
     * Return whether or not all vertices of the Triangulation are actually
     * inside the domain defined by the PatchHierarchy.
     */
    bool
    compute_vertices_inside_domain() const;

    /** @} */

  protected:
    /**
     * Reinitialize all the FE data structures, including vectors and mappings.
     */
    void
    reinit_dofs();

    /**
     * Reinitialize centroid data.
     */
    void
    reinit_centroid();

    /**
     * Reinitialize the NodalInteraction object.
     *
     * @note This function should typically be called after reinit_tria().
     */
    void
    reinit_interaction();

    /**
     * Helper function which calls the previous three functions in the correct
     * order (dofs, centroid, then interaction).
     *
     * Since inheriting classes set up meter_tria in a variety of different
     * ways, they should typically set up that object themselves first and then
     * call this function afterwards to manage the rest of the meter's state.
     */
    void
    internal_reinit();

    /**
     * Meter centroid.
     */
    Point<spacedim> centroid;

    /**
     * Meter centroid, in reference cell coordinates.
     */
    Point<dim> ref_centroid;

    /**
     * Cell containing the centroid.
     */
    typename Triangulation<dim, spacedim>::active_cell_iterator centroid_cell;

    /**
     * Cartesian-grid data.
     */
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy;

    /**
     * Meter Triangulation.
     */
    parallel::shared::Triangulation<dim, spacedim> meter_tria;

    /**
     * Mapping on the meter Triangulation.
     */
    std::unique_ptr<Mapping<dim, spacedim>> meter_mapping;

    /**
     * Quadrature to use on the meter mesh. Has degree $2 * scalar_fe.degree +
     * 1$.
     */
    Quadrature<dim> meter_quadrature;

    /**
     * Scalar FiniteElement used on meter_tria
     */
    std::unique_ptr<FiniteElement<dim, spacedim>> scalar_fe;

    /**
     * Vector FiniteElement used on meter_tria
     */
    std::unique_ptr<FiniteElement<dim, spacedim>> vector_fe;

    /**
     * DoFHandler for scalar quantities defined on meter_tria.
     */
    DoFHandler<dim, spacedim> scalar_dof_handler;

    /**
     * DoFHandler for vector-valued quantities defined on meter_tria.
     */
    DoFHandler<dim, spacedim> vector_dof_handler;

    std::shared_ptr<Utilities::MPI::Partitioner> vector_partitioner;

    std::shared_ptr<Utilities::MPI::Partitioner> scalar_partitioner;

    /**
     * Positions of the mesh DoFs - always the identity function after
     * reinitalization.
     */
    LinearAlgebra::distributed::Vector<double> identity_position;

    /**
     * Interaction object.
     */
    std::unique_ptr<NodalInteraction<dim, spacedim>> nodal_interaction;
  };


  // --------------------------- inline functions --------------------------- //


  template <int dim, int spacedim>
  inline const Triangulation<dim, spacedim> &
  MeterBase<dim, spacedim>::get_triangulation() const
  {
    return meter_tria;
  }

  template <int dim, int spacedim>
  inline const Mapping<dim, spacedim> &
  MeterBase<dim, spacedim>::get_mapping() const
  {
    return *meter_mapping;
  }

  template <int dim, int spacedim>
  inline const DoFHandler<dim, spacedim> &
  MeterBase<dim, spacedim>::get_scalar_dof_handler() const
  {
    return scalar_dof_handler;
  }

  template <int dim, int spacedim>
  inline const DoFHandler<dim, spacedim> &
  MeterBase<dim, spacedim>::get_vector_dof_handler() const
  {
    return vector_dof_handler;
  }

  template <int dim, int spacedim>
  Point<spacedim>
  MeterBase<dim, spacedim>::get_centroid() const
  {
    return centroid;
  }
} // namespace fdl

#endif
