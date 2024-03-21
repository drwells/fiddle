#ifndef included_fiddle_grid_intersection_map_h
#define included_fiddle_grid_intersection_map_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <fiddle/grid/patch_map.h>

#include <deal.II/base/linear_index_iterator.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <CellIndex.h>
#include <Patch.h>
#include <SideIndex.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

#include <iterator>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  namespace internal
  {
    /**
     * Storage of intersections between a single Patch and the displaced
     * elements. This class only stores single intersections: i.e., it assumes
     * each stored stencil only intersects a mesh exactly once.
     *
     * The intersections are stored in a compressed format:
     *
     * 1. As intersections are defined as pairs of adjacent cell indices, it
     *    suffices to store the lower index and the axis. To get the upper index
     *    we just increment the entry of the lower index corresponding to its
     *    axis.
     *
     * 2. A CellIndex, dx (grid spacing), and domain_x_lower together uniquely
     *    define the Cartesian coordinates of a point. Hence we store the
     *    Cartesian coordinates of the intersections themselves as convex
     *    combinations of the lower and upper points.
     *
     * 3. In the usual deal.II way we can construct Triangulation or DoFHandler
     *    iterators on-the-fly via the TriaIterator constructor (which only
     *    requires pointers to the Triangulation and DoFHandler, the cell level,
     *    and the cell index).
     *
     * As the maximum value for an axis is 2 and the maximum possible value for
     * a level is perhaps 20-30 (as the number of cells scales exponentially
     * with the level) it suffices to store both as unsigned chars to save a lot
     * of memory.
     *
     * As the name implies, this is only presently implemented for stencils
     * which intersect the FE mesh exactly once. Handling multiple intersections
     * is a future project.
     */
    template <int spacedim>
    struct PatchSingleIntersections
    {
      // Eulerian grid spacing.
      Tensor<1, spacedim> m_dx;

      // Bottom left corner of the domain.
      Point<spacedim> m_domain_x_lower;

      // lower index of each intersection.
      std::vector<pdat::CellIndex<spacedim>> m_lower_indices;

      // axis of each intersection.
      std::vector<unsigned char> m_axes;

      // convex combination coefficient of each intersection.
      std::vector<double> m_convex_coefficients;

      // deal.II cell level.
      std::vector<unsigned char> m_cell_level;

      // deal.II cell index.
      std::vector<int> m_cell_index;
    };
  } // namespace internal

  /**
   * Mapping between patches, elements, and the points at which finite
   * difference stencils intersect elements.
   *
   * @note This class presently only supports cell and side centered data as
   * those data centerings produce the same intersections.
   */
  template <int dim, int spacedim = dim>
  class PatchIntersectionMap
  {
  public:
    static_assert(dim + 1 == spacedim, "Only implemented for codim = 1");

    class Accessor;
    class Iterator;

    /**
     * Default constructor. Sets up an empty mapping.
     */
    PatchIntersectionMap() = default;

    /**
     * Proxy class granting access to an intersection.
     *
     * In fiddle, intersections are always defined with pairs of indices: i.e.,
     * stencils between two cell centers or (for, e.g., side-centered data)
     * stencils across a single cell.
     */
    class Accessor
    {
    public:
      /**
       * Type of pointer to the container. Required by LinearIndexIterator.
       */
      using container_pointer_type =
        const internal::PatchSingleIntersections<spacedim> *;

      /**
       * Size type. Required by LinearIndexIterator.
       */
      using size_type = std::size_t;

      /**
       * Constructor.
       */
      Accessor(const container_pointer_type container,
               const std::ptrdiff_t         index);

      /**
       * Constructor.
       */
      Accessor();

      /**
       * Return the lower cell intersection index.
       */
      pdat::CellIndex<spacedim>
      get_cell_lower() const;

      /**
       * Return the upper cell intersection index.
       */
      pdat::CellIndex<spacedim>
      get_cell_upper() const;

      /**
       * Return the lower side intersection index.
       */
      pdat::SideIndex<spacedim>
      get_side_lower() const;

      /**
       * Return the upper side intersection index.
       */
      pdat::SideIndex<spacedim>
      get_side_upper() const;

      /**
       * Return the axis of the intersection.
       *
       * @note This is equivalent to get_side_lower().getAxis().
       */
      int
      get_axis() const;

      /**
       * Return the convex combination coefficient defining the location of the
       * point between the lower and upper CellIndex.
       */
      double
      get_cell_convex_coefficient() const;

      /**
       * Return the convex combination coefficient defining the location of the
       * point between the lower and upper SideIndex.
       */
      double
      get_side_convex_coefficient() const;

      /**
       * Return the actual point of the intersection.
       */
      Point<spacedim>
      get_point() const;

    protected:
      void
      assert_valid() const;

      container_pointer_type container;

      std::ptrdiff_t linear_index;

      friend class LinearIndexIterator<Iterator, Accessor>;
    };

    class Iterator : public LinearIndexIterator<Iterator, Accessor>
    {
    public:
      Iterator(const internal::PatchSingleIntersections<spacedim> *container,
               const std::ptrdiff_t                                index);
    };

  protected:
    PatchMap<dim, spacedim> patch_map;
  };


  // --------------------------- inline functions --------------------------- //


  template <int dim, int spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::Accessor(
    const internal::PatchSingleIntersections<spacedim> *container,
    const std::ptrdiff_t                                index)
    : container(container)
    , linear_index(index)
  {}



  template <int dim, int spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::Accessor()
    : container(nullptr)
    , linear_index(std::numeric_limits<std::ptrdiff_t>::max())
  {}



  template <int dim, int spacedim>
  void
  PatchIntersectionMap<dim, spacedim>::Accessor::assert_valid() const
  {
    Assert(container, ExcMessage("The pointer should be set."));
    AssertIndexRange(linear_index, container->m_lower_indices.size());

    // Also verify PatchSingleIntersections
    AssertDimension(container->m_lower_indices.size(),
                    container->m_axes.size());
    AssertDimension(container->m_lower_indices.size(),
                    container->m_convex_coefficients.size());
    AssertDimension(container->m_lower_indices.size(),
                    container->m_cell_level.size());
    AssertDimension(container->m_lower_indices.size(),
                    container->m_cell_index.size());

    // Check that the convex combination is, in fact, convex
    const auto convex = container->m_convex_coefficients[linear_index];
    (void)convex;
    Assert(0.0 <= convex && convex <= 1.0,
           ExcMessage(
             "The convex combination coefficient should be between 0 and 1."));
  }



  template <int dim, int spacedim>
  pdat::CellIndex<spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::get_cell_lower() const
  {
    assert_valid();

    return container->m_lower_indices[linear_index];
  }



  template <int dim, int spacedim>
  pdat::CellIndex<spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::get_cell_upper() const
  {
    assert_valid();

    auto cell_index = container->m_lower_indices[linear_index];
    ++cell_index(int(container->m_axes[linear_index]));

    return cell_index;
  }



  template <int dim, int spacedim>
  pdat::SideIndex<spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::get_side_lower() const
  {
    assert_valid();

    // Like in IBAMR, if a point is on the interface between two cells we assign
    // it to the higher-index cell.
    const bool in_lower_cell = get_cell_convex_coefficient() < 0.5;

    return pdat::SideIndex<spacedim>(in_lower_cell ? get_cell_lower() :
                                                     get_cell_upper(),
                                     get_axis(),
                                     pdat::SideIndex<spacedim>::Lower);
  }



  template <int dim, int spacedim>
  pdat::SideIndex<spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::get_side_upper() const
  {
    assert_valid();

    const bool in_lower_cell =
      container->m_convex_coefficients[linear_index] < 0.5;

    return pdat::SideIndex<spacedim>(in_lower_cell ? get_cell_lower() :
                                                     get_cell_upper(),
                                     get_axis(),
                                     pdat::SideIndex<spacedim>::Upper);
  }



  template <int dim, int spacedim>
  int
  PatchIntersectionMap<dim, spacedim>::Accessor::get_axis() const
  {
    assert_valid();

    return container->m_axes[linear_index];
  }



  template <int dim, int spacedim>
  double
  PatchIntersectionMap<dim, spacedim>::Accessor::get_cell_convex_coefficient()
    const
  {
    assert_valid();

    return container->m_convex_coefficients[linear_index];
  }



  template <int dim, int spacedim>
  double
  PatchIntersectionMap<dim, spacedim>::Accessor::get_side_convex_coefficient()
    const
  {
    assert_valid();

    const double cell_convex   = get_cell_convex_coefficient();
    const bool   in_lower_cell = cell_convex < 0.5;

    if (in_lower_cell)
      {
        return cell_convex + 0.5;
      }
    else
      {
        return cell_convex - 0.5;
      }
  }



  template <int dim, int spacedim>
  Point<spacedim>
  PatchIntersectionMap<dim, spacedim>::Accessor::get_point() const
  {
    assert_valid();

    const auto      cell_index = get_cell_lower();
    const auto      dx         = container->m_dx;
    Point<spacedim> result     = container->m_domain_x_lower;

    for (unsigned int d = 0; d < spacedim; ++d)
      result[d] += (double(cell_index(d)) + 0.5) * dx[d];

    result[get_axis()] += dx[get_axis()] * get_cell_convex_coefficient();

    return result;
  }



  template <int dim, int spacedim>
  PatchIntersectionMap<dim, spacedim>::Iterator::Iterator(
    const internal::PatchSingleIntersections<spacedim> *container,
    const std::ptrdiff_t                                index)
    : LinearIndexIterator<Iterator, Accessor>(
        PatchIntersectionMap<dim, spacedim>::Accessor(container, index))
  {}
} // namespace fdl

#endif
