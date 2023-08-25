#include <fiddle/grid/patch_intersection_map.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IndexUtilities.h>

#include <CartesianGridGeometry.h>

// Test PatchIntersectionMap::Accessor's SAMRAI index calculations

using namespace SAMRAI;
using namespace dealii;

int
main(int argc, char **argv)
{
  const auto     mpi_comm = MPI_COMM_WORLD;
  IBTK::IBTKInit ibtk_init(argc, argv, mpi_comm);

  tbox::Pointer<IBTK::AppInitializer> app_initializer =
    new IBTK::AppInitializer(argc, argv, "logfile");
  tbox::Pointer<tbox::Database> input_db = app_initializer->getInputDatabase();

  fdl::internal::PatchSingleIntersections<NDIM> intersections;

  tbox::Pointer<geom::CartesianGridGeometry<NDIM>> grid_geometry =
    new geom::CartesianGridGeometry<NDIM>("CartesianGeometry",
                                          app_initializer->getComponentDatabase(
                                            "CartesianGeometry"));

  for (unsigned int d = 0; d < NDIM; ++d)
    {
      intersections.dx[d]             = grid_geometry->getDx()[d];
      intersections.domain_x_lower[d] = grid_geometry->getXLower()[d];
    }

  intersections.lower_indices.emplace_back(hier::Index<NDIM>(0, 0));
  intersections.lower_indices.emplace_back(hier::Index<NDIM>(0, 0));
  intersections.lower_indices.emplace_back(hier::Index<NDIM>(1, 0));

  intersections.axes.emplace_back(0);
  intersections.axes.emplace_back(1);
  intersections.axes.emplace_back(0);

  intersections.convex_coefficients.emplace_back(0.0);
  intersections.convex_coefficients.emplace_back(0.25);
  intersections.convex_coefficients.emplace_back(1.0);

  // Not used in the test but we need to allocate them anyway (there is an
  // assertion that all of these arrays have the same length)
  intersections.cell_level.push_back(0);
  intersections.cell_level.push_back(0);
  intersections.cell_level.push_back(0);

  intersections.cell_index.push_back(0);
  intersections.cell_index.push_back(1);
  intersections.cell_index.push_back(0);

  fdl::PatchIntersectionMap<NDIM - 1, NDIM>::Iterator it(&intersections, 0);

  for (unsigned int i = 0; i < intersections.lower_indices.size(); ++i)
    {
      const pdat::SideIndex<NDIM> side_lower_index = it->get_side_lower();
      const pdat::SideIndex<NDIM> side_upper_index = it->get_side_upper();

      tbox::pout << "index = " << i << '\n'
                 << "  Cell lower = " << it->get_cell_lower() << '\n'
                 << "  Cell upper = " << it->get_cell_upper() << '\n'
                 << "  Side lower = " << side_lower_index
                 << " axis = " << side_lower_index.getAxis() << '\n'
                 << "  Side upper = " << side_upper_index
                 << " axis = " << side_upper_index.getAxis() << '\n'
                 << "  axis = " << it->get_axis() << '\n'
                 << "  cell convex = " << it->get_cell_convex_coefficient()
                 << '\n'
                 << "  side convex = " << it->get_side_convex_coefficient()
                 << '\n'
                 << "  point = " << it->get_point() << '\n';


      hier::Index<NDIM>  ratio(1);
      const IBTK::Vector side_lower =
        IBTK::IndexUtilities::getSideCenter(grid_geometry,
                                            ratio,
                                            side_lower_index);
      const IBTK::Vector side_upper =
        IBTK::IndexUtilities::getSideCenter(grid_geometry,
                                            ratio,
                                            side_upper_index);

      tbox::pout << "  computed side lower = " << side_lower.transpose() << '\n'
                 << "  computed side upper = " << side_upper.transpose()
                 << '\n';

      ++it;
    }
}
