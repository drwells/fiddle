#include <fiddle/base/initial_guess.h>

#include <deal.II/lac/vector.h>

#include <fstream>

// Essentially a copy-and-paste of the IBTK test I wrote: I hold copyright so I
// can do that

int
main()
{
  std::ofstream out("output");

  using namespace dealii;

  // Make sure that we project onto a single vector
  {
    fdl::InitialGuess<Vector<double>> guess(1);
    Vector<double>                    solution(3);
    Vector<double>                    rhs(3);

    solution[0] = 1.0;
    solution[1] = 1.0;
    solution[2] = 1.0;

    rhs[0] = 1.0;
    rhs[1] = 1.0;

    guess.submit(solution, rhs);

    Vector<double> new_rhs(3);
    new_rhs[0] = 1.0;
    guess.guess(solution, new_rhs);

    solution.print(out);
  }

  // Same but use the same vector over and over again (i.e., with a singular
  // projection matrix)
  {
    fdl::InitialGuess<Vector<double>> guess(3);

    for (unsigned int i = 0; i < 10; ++i)
      {
        Vector<double> solution(3);
        Vector<double> rhs(3);

        solution[0] = 1.0;
        solution[1] = 1.0;
        solution[2] = 1.0;

        rhs[0] = 1.0;
        rhs[1] = 1.0;

        guess.submit(solution, rhs);
      }

    Vector<double> new_rhs(3);
    new_rhs[0] = 1.0;
    Vector<double> solution(3);
    guess.guess(solution, new_rhs);

    solution.print(out);
  }

  // Check with two vectors
  {
    fdl::InitialGuess<Vector<double>> guess(10);

    Vector<double> solution_1(3);
    Vector<double> rhs_1(3);

    solution_1[0] = 1.0;
    solution_1[1] = 1.0;
    solution_1[2] = 1.0;

    rhs_1[0] = 1.0;
    rhs_1[1] = 1.0;
    guess.submit(solution_1, rhs_1);

    Vector<double> solution_2(3);
    Vector<double> rhs_2(3);

    solution_2[0] = 1.0;
    solution_2[1] = 2.0;
    solution_2[2] = 3.0;

    rhs_2[0] = 1.0;
    rhs_2[1] = -1.0;

    guess.submit(solution_2, rhs_2);

    Vector<double> new_rhs(3);
    new_rhs[0] = 1.0;
    Vector<double> solution(3);
    guess.guess(solution, new_rhs);
    solution.print(out);

    // shouldn't change the vector
    guess.submit(solution, new_rhs);
    guess.guess(solution, new_rhs);
    solution.print(out);
  }

  // Do the same thing but duplicate the vectors
  {
    fdl::InitialGuess<Vector<double>> guess(10);

    for (unsigned int i = 0; i < 10; ++i)
      {
        Vector<double> solution_1(3);
        Vector<double> rhs_1(3);

        solution_1[0] = 1;
        solution_1[1] = 1;
        solution_1[2] = 1;

        rhs_1[0] = 1;
        rhs_1[1] = 1;
        guess.submit(solution_1, rhs_1);
      }

    for (unsigned int i = 0; i < 9; ++i)
      {
        Vector<double> solution_2(3);
        Vector<double> rhs_2(3);

        solution_2[0] = 1;
        solution_2[1] = 2;
        solution_2[2] = 3;

        rhs_2[0] = 1;
        rhs_2[1] = -1;

        guess.submit(solution_2, rhs_2);
      }

    Vector<double> new_rhs(3);
    new_rhs[0] = 1;
    Vector<double> solution(3);
    guess.guess(solution, new_rhs);
    solution.print(out);

    // shouldn't change the solution
    guess.submit(solution, new_rhs);
    guess.guess(solution, new_rhs);

    solution.print(out);
  }

  // Do something triangular
  {
    fdl::InitialGuess<Vector<double>> guess(10);

    for (unsigned int i = 0; i < 5; ++i)
      {
        Vector<double> solution(10);
        Vector<double> rhs(10);

        for (unsigned int j = 0; j < i; ++j)
          {
            solution[j] = j;
            rhs[j]      = j;
          }

        guess.submit(solution, rhs);
      }

    Vector<double> new_rhs(10);
    new_rhs[0] = 1;
    new_rhs[1] = 2;
    new_rhs[2] = 3;
    new_rhs[3] = 4;
    new_rhs[4] = 5;
    Vector<double> solution(10);
    guess.guess(solution, new_rhs);

    solution.print(out);
  }
}
