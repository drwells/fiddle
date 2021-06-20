#include <fiddle/base/quadrature_family.h>

#include <fstream>

// Print out the maximum point distances.

template <int dim>
void
test_max_point_distances(const fdl::QGaussFamily<dim> &q_family,
                         std::ofstream &               out)
{
  // populate cached entries
  for (unsigned char i = 0; i < 10; ++i)
    q_family[i];

  // and print what we found
  const auto max_point_distances = q_family.get_max_point_distances();
  for (unsigned char i = 0; i < max_point_distances.size(); ++i)
    out << "i = " << int(i) << " n_points = " << q_family[i].size()
        << " max distance = " << max_point_distances[i] << '\n';
}

int
main()
{
  std::ofstream out("output");
  for (unsigned int min_points_1D = 1; min_points_1D < 5; ++min_points_1D)
    {
      out << "min points 1D " << min_points_1D << "\n";
      fdl::QGaussFamily<2> q_family(min_points_1D);
      test_max_point_distances(q_family, out);
    }
}
