#ifndef included_fiddle_initial_guess_h
#define included_fiddle_initial_guess_h

#include <fiddle/base/config.h>

#include <ibtk/config.h>

IBTK_DISABLE_EXTRA_WARNINGS
#include <Eigen/Core>
IBTK_ENABLE_EXTRA_WARNINGS

#include <deque>

namespace fdl
{
  /**
   * Class for computing initial guesses - essentially the same as
   * IBTK::InitialGuess. Uses the 'Fischer-3' algorithm (same as PETSc) to
   * compute guesses via projection.
   */
  template <typename VectorType>
  class InitialGuess
  {
  public:
    explicit InitialGuess(const unsigned int n_vectors = 5);

    void
    submit(const VectorType &solution, const VectorType &rhs);

    void
    guess(VectorType &solution, const VectorType &rhs);

  protected:
    unsigned int n_max_vectors;
    unsigned int n_stored_vectors;

    // IBAMR always has Eigen, deal.II only optionally has LAPACK
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> correlation_matrix;

    Eigen::VectorXd projection_coefficients;

    const VectorType *last_rhs;

    std::deque<VectorType> solutions;
    std::deque<VectorType> right_hand_sides;
  };
} // namespace fdl
#endif
