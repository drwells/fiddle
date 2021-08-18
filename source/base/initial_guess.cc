#include <fiddle/base/exceptions.h>
#include <fiddle/base/initial_guess.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <Eigen/Dense>

namespace fdl
{
  using namespace dealii;

  template <typename VectorType>
  InitialGuess<VectorType>::InitialGuess(const unsigned int n_vectors)
    : n_max_vectors(n_vectors)
    , n_stored_vectors(0)
    , last_rhs(nullptr)
  {}

  template <typename VectorType>
  void
  InitialGuess<VectorType>::submit(const VectorType &solution,
                                   const VectorType &rhs)
  {
    if (n_max_vectors == 0)
      return;
    // update our list of vectors:
    if (n_stored_vectors == n_max_vectors)
      {
        // oldest entry is first
        solutions.erase(solutions.begin());
        solutions.push_back(solution);
        right_hand_sides.erase(right_hand_sides.begin());
        right_hand_sides.push_back(rhs);

        // shift the computed dot products up and to the left:
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat_copy(
          correlation_matrix);
        for (unsigned int i = 1; i < n_max_vectors; ++i)
          {
            for (unsigned int j = 1; j < n_max_vectors; ++j)
              {
                correlation_matrix(i - 1, j - 1) = mat_copy(i, j);
              }
          }

        // Recycle dot products if we can. The indices are offset by one since
        // we removed the oldest vector.
        if (&rhs == last_rhs)
          {
            for (unsigned int i = 0; i < n_max_vectors - 1; ++i)
              {
                correlation_matrix(n_max_vectors - 1, i) =
                  projection_coefficients[i + 1];
                correlation_matrix(i, n_max_vectors - 1) =
                  projection_coefficients[i + 1];
#ifdef DEBUG
                const auto new_inner = right_hand_sides[i] * rhs;
                Assert(std::abs(new_inner - projection_coefficients[i + 1]) <=
                         1e-14 * std::abs(projection_coefficients[i + 1]),
                       ExcMessage("This class assumes that the RHS vectors are "
                                  "not modified between calls."));
#endif
              }
            correlation_matrix(n_max_vectors - 1, n_max_vectors - 1) =
              rhs * rhs;
            return;
          }
      }
    else
      {
        ++n_stored_vectors;
        solutions.push_back(solution);
        right_hand_sides.push_back(rhs);

        // Save the prior dot products in a different way:
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat_copy;

        mat_copy = correlation_matrix;
        correlation_matrix.resize(n_stored_vectors, n_stored_vectors);
        for (unsigned int i = 0; i < n_stored_vectors - 1; ++i)
          {
            for (unsigned int j = 0; j < n_stored_vectors - 1; ++j)
              {
                correlation_matrix(i, j) = mat_copy(i, j);
              }
          }

        // We kept the oldest vector so there is no offset in the copy
        if (&rhs == last_rhs)
          {
            for (unsigned int i = 0; i < n_stored_vectors - 1; ++i)
              {
                correlation_matrix(n_stored_vectors - 1, i) =
                  projection_coefficients[i];
                correlation_matrix(i, n_stored_vectors - 1) =
                  projection_coefficients[i];
#ifdef DEBUG
                const auto new_inner = right_hand_sides[i] * rhs;
                Assert(std::abs(new_inner - projection_coefficients[i]) <=
                         1e-14 * std::abs(projection_coefficients[i]),
                       ExcMessage("This class assumes that the RHS vectors are "
                                  "not modified between calls."));
#endif
              }
            correlation_matrix(n_stored_vectors - 1, n_stored_vectors - 1) =
              rhs * rhs;
            return;
          }
      }

    Assert(last_rhs == nullptr, ExcFDLInternalError());
    // Compute the last row and then copy it into the last column.
    for (unsigned int j = 0; j < n_stored_vectors; ++j)
      {
        const double inner = right_hand_sides.back() * right_hand_sides[j];
        correlation_matrix(n_stored_vectors - 1, j) = inner;
        correlation_matrix(j, n_stored_vectors - 1) = inner;
      }
  }

  template <typename VectorType>
  void
  InitialGuess<VectorType>::guess(VectorType &solution, const VectorType &rhs)
  {
    if (n_stored_vectors == 0)
      {
        return;
      }

    projection_coefficients.resize(n_stored_vectors, 1);
    for (unsigned int i = 0; i < n_stored_vectors; ++i)
      {
        projection_coefficients[i] = right_hand_sides[i] * rhs;
      }
    last_rhs = &rhs;

    Eigen::VectorXd coefs(n_stored_vectors);
    // Should the SVD fail for any reason just use the last solution as a guess.
    try
      {
        // SVD doesn't make sense with invalid input - throw something and
        // immediately catch it
        if (!correlation_matrix.allFinite())
          throw int();
        if (!projection_coefficients.allFinite())
          throw int();
        Eigen::JacobiSVD<decltype(correlation_matrix)> svd(
          correlation_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        coefs = svd.solve(projection_coefficients);
      }
    catch (...)
      {
        coefs.fill(0.0);
        coefs(n_stored_vectors - 1) = 1.0;
      }

    solution = 0.0;
    for (unsigned int i = 0; i < n_stored_vectors; ++i)
      {
        solution.add(coefs[i], solutions[i]);
      }
  }

  template class InitialGuess<Vector<float>>;
  template class InitialGuess<Vector<double>>;

  template class InitialGuess<LinearAlgebra::distributed::Vector<float>>;
  template class InitialGuess<LinearAlgebra::distributed::Vector<double>>;

} // namespace fdl
