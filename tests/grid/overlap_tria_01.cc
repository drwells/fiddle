#include <fiddle/grid/intersection_predicate.h>
#include <fiddle/grid/overlap_tria.h>

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/grid_generator.h>

#include <fstream>

// verify that we can set up an 'empty' overlap tria. It will always have one
// cell.

class NoIntersections : public fdl::IntersectionPredicate<2>
{
  virtual bool
  operator()(
    const dealii::Triangulation<2>::cell_iterator & /*cell*/) const override
  {
    return false;
  }
};

int
main(int argc, char **argv)
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize   mpi_initialization(argc, argv, 1);
  parallel::shared::Triangulation<2> shared_tria(MPI_COMM_WORLD);

  GridGenerator::hyper_ball(shared_tria);

  fdl::OverlapTriangulation<2> overlap_tria(shared_tria, NoIntersections());

  std::ofstream out("output");

  out << "Number of active cells = " << overlap_tria.n_active_cells() << '\n';
}
