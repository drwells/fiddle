#ifndef included_fiddle_initial_guess_h
#define included_fiddle_initial_guess_h

#include <ibtk/config.h>

IBTK_DISABLE_EXTRA_WARNINGS
#include <Eigen/Core>
IBTK_ENABLE_EXTRA_WARNINGS

#include <deque>

namespace fdl
{
    template <typename VectorType>
    class InitialGuess
    {
    public:
      InitialGuess(const unsigned int n_vectors = 5);

      void
      submit(const VectorType &solution, const VectorType &rhs);

      void
      guess(VectorType &solution, const VectorType &rhs) const;

    protected:
      unsigned int n_max_vectors;
      unsigned int n_stored_vectors;

      // IBAMR always has Eigen, deal.II only optionally has LAPACK
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> correlation_matrix;

      std::deque<VectorType> solutions;
      std::deque<VectorType> right_hand_sides;
    };
}
#endif
