#include <deal.II/numerics/vector_tools_evaluate.h>

#include <fiddle/postprocess/point_values.h>

namespace fdl
{
  using namespace dealii;

  namespace
  {
    std::vector<Tensor<1, 1>>
    convert(std::vector<double> &input)
    {
      std::vector<Tensor<1, 1>> result(input.size());
      for (std::size_t i = 0; i < input.size(); ++i)
        result[i][0] = input[i];

      return result;
    }

    template <int dim>
    std::vector<Tensor<1, dim>>
    convert(std::vector<Tensor<1, dim>> &input)
    {
      return std::move(input);
    }
  } // namespace

  template <int n_components, int dim, int spacedim>
  PointValues<n_components, dim, spacedim>::PointValues(
    const Mapping<dim, spacedim> &      mapping,
    const DoFHandler<dim, spacedim> &   dof_handler,
    const std::vector<Point<spacedim>> &evaluation_points)
    : mapping(&mapping)
    , dof_handler(&dof_handler)
    , evaluation_points(evaluation_points)
  {}

  template <int n_components, int dim, int spacedim>
  std::vector<Tensor<1, n_components>>
  PointValues<n_components, dim, spacedim>::evaluate(
    const LinearAlgebra::distributed::Vector<double> &vector) const
  {
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    auto result =
      VectorTools::point_values<n_components, dim, spacedim, VectorType>(
        *mapping,
        *dof_handler,
        vector,
        evaluation_points,
        remote_point_evaluation);

    return convert(result);
  }


  // TODO - RemotePointEvaluation doesn't work with codim 1 yet
  // template class PointValues<1, NDIM - 1, NDIM>;
  // template class PointValues<NDIM, NDIM - 1, NDIM>;
  template class PointValues<1, NDIM, NDIM>;
  template class PointValues<NDIM, NDIM, NDIM>;
} // namespace fdl
