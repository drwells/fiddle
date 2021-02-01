#ifndef included_fiddle_grid_patch_map_h
#define included_fiddle_grid_patch_map_h

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/intersection_predicate.h>

#include <deal.II/dofs/dof_handler.h>

#include <Patch.h>

namespace fdl
{
  using namespace dealii;
  /**
   * Mapping between patches and elements. Assumes that the Triangulation
   * associated with the position DoFHandler already intersects the patches in
   * some meaningful way.
   */
  template <int dim, int spacedim = dim>
  class PatchMap
  {
  public:
    /**
     * Default constructor. Sets up an empty mapping.
     */
    PatchMap() = default;

    /**
     * Constructor. Associates cells to patches from provided bounding boxes.
     */
    template <typename Number>
    PatchMap(
      const std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>>
        &                                 patches,
      const double                        extra_ghost_cell_fraction,
      const Triangulation<dim, spacedim> &tria,
      std::vector<BoundingBox<spacedim, Number>> &cell_bboxes);

    /**
     * Return the number of patches.
     */
    std::size_t
    size() const;

    /**
     * Iterator class for looping over cells of a DoFHandler corresponding to
     * the stored triangulation.
     */
    class iterator
    {
    public:
      using value_type =
        typename DoFHandler<dim, spacedim>::active_cell_iterator;

      using difference_type = std::ptrdiff_t;

      using reference = value_type &;

      using size_type = std::size_t;

      value_type
      operator*() const;

      // basic iterator arithmetic
      iterator &
      operator+=(const difference_type n);

      iterator
      operator++();

      iterator
      operator++(int);

      iterator
      operator+(const difference_type n) const;

      iterator &
      operator-=(const difference_type n);

      iterator
      operator--();

      iterator
      operator--(int);

      iterator
      operator-(const difference_type n) const;

      difference_type
      operator-(const iterator &other) const;

      // comparisons
      bool
      operator<(const iterator &other) const;

      bool
      operator==(const iterator &other) const;

      bool
      operator!=(const iterator &other) const;

    protected:
      // only let a PatchMap construct these iterators directly
      iterator(const std::ptrdiff_t             index,
               const DoFHandler<dim, spacedim> &dof_handler,
               const std::vector<
                 typename Triangulation<dim, spacedim>::active_cell_iterator>
                 &patch_cells)
        : dh(&dof_handler)
        , cells(&patch_cells)
        , index(index)
      {}

      const DoFHandler<dim, spacedim> *dh;

      const std::vector<
        typename Triangulation<dim, spacedim>::active_cell_iterator> *cells;

      std::ptrdiff_t index;

      template <int, int>
      friend class PatchMap;
    };

    SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>> &
    get_patch(const std::size_t patch_n)
    {
      AssertIndexRange(patch_n, size());
      return patches[patch_n];
    }

    const SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>> &
    get_patch(const std::size_t patch_n) const
    {
      AssertIndexRange(patch_n, size());
      return patches[patch_n];
    }

    iterator
    begin(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const
    {
      AssertIndexRange(patch_n, size());
      return iterator(0, dh, cells[patch_n]);
    }

    iterator
    end(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const
    {
      AssertIndexRange(patch_n, size());
      return iterator(cells[patch_n].size(), dh, cells[patch_n]);
    }

  protected:
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<spacedim>>> patches;

    // TODO - we can really compress this down by instead storing
    //
    // std::vector<std::vector<IndexSet>>
    //
    // where the first index is the patch number, the second is the level
    // number, and the IndexSet stores the cell indices.
    std::vector<
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
      cells;
  };

  // ---------------------------- inline functions -----------------------------

  template <int dim, int spacedim>
  std::size_t
  PatchMap<dim, spacedim>::size() const
  {
    return patches.size();
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator::value_type
  PatchMap<dim, spacedim>::iterator::operator*() const
  {
    Assert(0 <= index && index <= cells->size(),
           ExcMessage("invalid iterator"));
    if (index == cells->size())
      return dh->end();

    return typename DoFHandler<dim, spacedim>::active_cell_iterator(
      &dh->get_triangulation(),
      (*cells)[index]->level(),
      (*cells)[index]->index(),
      dh);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator &
  PatchMap<dim, spacedim>::iterator::operator+=(const difference_type n)
  {
    index += n;
    return *this;
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator++()
  {
    return operator+=(1);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator++(int)
  {
    auto  copy = *this;
    this->operator+=(1);
    return copy;
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator+(const difference_type n) const
  {
    auto copy = *this;
    copy += n;
    return copy;
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator &
  PatchMap<dim, spacedim>::iterator::operator-=(const difference_type n)
  {
    return operator+=(-n);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator--()
  {
    return operator-=(1);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator--(int)
  {
    auto  copy = *this;
    this->operator-=(1);
    return copy;
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::iterator::operator-(const difference_type n) const
  {
    auto copy = *this;
    copy -= n;
    return copy;
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator::difference_type
  PatchMap<dim, spacedim>::iterator::operator-(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.cells == this->cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return index - other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator<(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.cells == this->cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index < other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator==(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.cells == this->cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index == other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator!=(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.cells == this->cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index != other.index;
  }
} // namespace fdl

#endif
