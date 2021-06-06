
#include <fiddle/base/utilities.h>

#include <algorithm>
#include <fstream>
#include <random>

using namespace dealii;

template <int spacedim>
void
compute_and_print(const std::vector<Point<spacedim>> &points, std::ostream &out)
{
  const auto pair = fdl::find_largest_nonintersecting_sphere(points);
  out << "diagonal point: " << pair.first
      << " diagonal diameter: " << pair.second << '\n';
}

template <int spacedim>
void
test(std::ostream &output)
{
  // test points on a diagonal
  std::vector<Point<spacedim>> diagonal_points;
  for (unsigned int i = 0; i < 10; ++i)
    {
      diagonal_points.emplace_back();
      for (unsigned int d = 0; d < spacedim; ++d)
        diagonal_points.back()[d] = i * i;
    }

  compute_and_print(diagonal_points, output);

  // do it again with a shuffle:
  {
    std::mt19937 gen(0);
    std::shuffle(diagonal_points.begin(), diagonal_points.end(), gen);
  }
  compute_and_print(diagonal_points, output);

  // Add points on a line:
  auto all_points = diagonal_points;
  for (unsigned int i = 0; i < 10; ++i)
    {
      all_points.emplace_back();
      all_points.back()[0] = i;
    }
  compute_and_print(all_points, output);

  // do it again with a shuffle:
  {
    std::mt19937 gen(0);
    std::shuffle(all_points.begin(), all_points.end(), gen);
  }
  compute_and_print(all_points, output);
}

int
main()
{
  std::ofstream output("output");
  test<NDIM>(output);
}
