#ifndef included_fiddle_postprocess_point_values_h
#define included_fiddle_postprocess_point_values_h

#include <fiddle/base/config.h>

FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <deal.II/base/mpi_remote_point_evaluation.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <vector>

namespace fdl
{
  using namespace dealii;

  /**
   * Convenience class for computing values of a finite element field at a set
   * of known points over time. Sets up some internal data structures that make
   * repeated calls to evaluate() much faster.
   */
  template <int n_components, int dim, int spacedim = dim>
  class PointValues
  {
  public:
    /**
     * Constructor. Using this class for the position itself requires some care.
     * In particular, @p mapping and @p evaluation_points should be in reference
     * coordinates (so @p mapping should usually be the one returned by
     * Part::get_mapping() and @p evaluation_points should be points in the
     * reference configuration, i.e., points inside the Triangulation itself).
     */
    PointValues(const Mapping<dim, spacedim>       &mapping,
                const DoFHandler<dim, spacedim>    &dof_handler,
                const std::vector<Point<spacedim>> &evaluation_points);

    /**
     * Evaluate the finite element field specified by @p vector at the stored
     * evaluation points. For example - to get the displacement of a point over
     * time, use the position vector here.
     */
    std::vector<Tensor<1, n_components>>
    evaluate(const LinearAlgebra::distributed::Vector<double> &vector) const;

    /**
     * Return a reference to the evaluation points originally used to set up
     * this object.
     */
    const std::vector<Point<spacedim>> &
    get_evaluation_points() const;

  protected:
    /**
     * Pointer to the provided mapping.
     */
    SmartPointer<const Mapping<dim, spacedim>> m_mapping;

    /**
     * Pointer to the provided DoFHandler.
     */
    SmartPointer<const DoFHandler<dim, spacedim>> m_dof_handler;

    /**
     * Points in reference coordinates where we will evaluate the finite element
     * field.
     */
    std::vector<Point<spacedim>> m_evaluation_points;

    /**
     * Internal data structure that manages both evaluation and communication.
     */
    Utilities::MPI::RemotePointEvaluation<dim, spacedim>
      m_remote_point_evaluation;
  };


  // --------------------------- inline functions --------------------------- //


  template <int n_components, int dim, int spacedim>
  const std::vector<Point<spacedim>> &
  PointValues<n_components, dim, spacedim>::get_evaluation_points() const
  {
    return m_evaluation_points;
  }
} // namespace fdl

#endif
