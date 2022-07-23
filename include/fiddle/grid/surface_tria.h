#ifndef included_fiddle_surface_tria_h
#define included_fiddle_surface_tria_h

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <limits>
#include <vector>

namespace fdl
{
  using namespace dealii;

  namespace Triangle
  {
    /**
     * Parameters for the call to Triangle.
     */
    struct AdditionalData
    {
      AdditionalData()
        : min_angle(30.0)
        , target_element_area(std::numeric_limits<double>::max())
      {}

      /**
       * Minimum angle in degrees. Large angles (e.g., 40) can cause the mesh
       * generator to create an unnecessarily large number of elements, so the
       * default value (or a lower one) is recommended.
       */
      double min_angle;

      /**
       * Target area. Defaults to elements with an edge length equal to the
       * distance between the first two nodes.
       */
      double target_element_area;
    };
  } // namespace Triangle

  /**
   * Triangulate a surface described by a convex hull.
   */
  void
  triangulate_convex(const std::vector<Point<2>>   &hull_vertices,
                     Triangulation<2>              &tria,
                     const Triangle::AdditionalData additional_data = {});

  /**
   * Set up a planar mesh which best fits (in the least-squares sense) the three
   * dimensional points.
   *
   * As the output mesh is planar, this algorithm first projects all points onto
   * a plane and then uses them as a convex hull.
   */
  Tensor<1, 3>
  setup_planar_meter_mesh(const std::vector<Point<3>>   &points,
                          Triangulation<2, 3>           &tria,
                          const Triangle::AdditionalData additional_data = {});

} // namespace fdl

#endif
