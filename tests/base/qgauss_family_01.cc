#include <fiddle/base/quadrature_family.h>

#include <fstream>

template <int dim>
void
test(const fdl::QuadratureFamily<dim> &q_family, std::ofstream &out)
{
  for (unsigned char i = 1; i < 6; ++i)
    {
      const dealii::Quadrature<dim> &quad = q_family[i];
      out << "quadrature " << int(i) << " size = " << quad.size() << '\n';
      for (const dealii::Point<dim> &point : quad.get_points())
        out << point << '\n';
    }
}

int
main()
{
  std::ofstream out("output");
  for (unsigned int min_points_1D = 1; min_points_1D < 5; ++min_points_1D)
    {
      out << "min points 1D " << min_points_1D << "\n";
      fdl::QGaussFamily<2> q_family(min_points_1D);
      test(q_family, out);
    }
}
