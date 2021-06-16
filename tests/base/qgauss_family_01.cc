#include <fiddle/base/quadrature_family.h>

#include <fstream>

template <int dim>
void
test_points(const fdl::QuadratureFamily<dim> &q_family, std::ofstream &out)
{
  for (unsigned char i = 0; i < 10; ++i)
    {
      const dealii::Quadrature<dim> &quad = q_family[i];
      out << "quadrature " << int(i) << " size = " << quad.size() << '\n';
      // We are never going to get a quadrature rule where the first few points
      // are identical and the rest change, so just print the first five or so
      for (unsigned int i = 0; i < std::min<unsigned int>(5, quad.size()); ++i)
        out << quad.get_points()[i] << '\n';
    }
}

template <int dim>
void
test_indices(const fdl::QuadratureFamily<dim> &q_family, std::ofstream &out)
{
  out << "dx = " << 0.1 << " DX = " << 0.05
      << " index = " << int(q_family.get_n_points_1D(0.1, 0.05)) << '\n';
  out << "dx = " << 0.1 << " DX = " << 0.1
      << " index = " << int(q_family.get_n_points_1D(0.1, 0.1)) << '\n';
  out << "dx = " << 0.1 << " DX = " << 0.2
      << " index = " << int(q_family.get_n_points_1D(0.1, 0.2)) << '\n';
  out << "dx = " << 0.1 << " DX = " << 0.4
      << " index = " << int(q_family.get_n_points_1D(0.1, 0.4)) << '\n';
  out << "dx = " << 0.1 << " DX = " << 0.8
      << " index = " << int(q_family.get_n_points_1D(0.1, 0.8)) << '\n';
}

int
main()
{
  std::ofstream out("output");
  for (unsigned int min_points_1D = 1; min_points_1D < 5; ++min_points_1D)
    {
      out << "min points 1D " << min_points_1D << "\n";
      fdl::QGaussFamily<2> q_family(min_points_1D);
      test_points(q_family, out);
      test_indices(q_family, out);
    }
}
