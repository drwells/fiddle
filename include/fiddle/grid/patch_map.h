#ifndef included_fiddle_grid_patch_map_h
#define included_fiddle_grid_patch_map_h

#include <fiddle/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>

#include <Patch.h>

#include <iterator>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

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
    PatchMap(const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
             const double                        extra_ghost_cell_fraction,
             const Triangulation<dim, spacedim> &tria,
             const std::vector<BoundingBox<spacedim, Number>> &cell_bboxes);

    /**
     * Same as the constructor.
     */
    template <typename Number>
    void
    reinit(const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
           const double                        extra_ghost_cell_fraction,
           const Triangulation<dim, spacedim> &tria,
           const std::vector<BoundingBox<spacedim, Number>> &cell_bboxes);


    /**
     * Return the number of patches.
     */
    std::size_t
    size() const;

    /**
     * Return a constant reference to the Triangulation.
     */
    const Triangulation<dim, spacedim> &
    get_triangulation() const;

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

      using iterator_category = std::random_access_iterator_tag;

      using pointer = value_type *;

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
               const std::vector<IndexSet> &    patch_cells)
        : dh(&dof_handler)
        , level_cells(&patch_cells)
        , index(index)
      {}

      const DoFHandler<dim, spacedim> *dh;

      const std::vector<IndexSet> *level_cells;

      std::ptrdiff_t index;

      template <int, int>
      friend class PatchMap;
    };

    /**
     * Get a patch stored by the patch map.
     *
     * @note Unlike SAMRAI's patch indexing, patch numbers here are local -
     * i.e., patch 42 on processor 0 is completely unrelated to patch 42 on
     * processor 1.
     */
    tbox::Pointer<hier::Patch<spacedim>> &
    get_patch(const std::size_t patch_n)
    {
      AssertIndexRange(patch_n, size());
      return patches[patch_n];
    }

    const tbox::Pointer<hier::Patch<spacedim>> &
    get_patch(const std::size_t patch_n) const
    {
      AssertIndexRange(patch_n, size());
      return patches[patch_n];
    }

    iterator
    begin(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const
    {
      AssertIndexRange(patch_n, size());
      Assert(&dh.get_triangulation() == &*tria,
             ExcMessage("must use same Triangulation"));
      return iterator(0, dh, patch_level_cells[patch_n]);
    }

    iterator
    end(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const
    {
      AssertIndexRange(patch_n, size());
      Assert(&dh.get_triangulation() == &*tria,
             ExcMessage("must use same Triangulation"));
      std::ptrdiff_t index = 0;
      for (const IndexSet &indices : patch_level_cells[patch_n])
        index += indices.n_elements();
      return iterator(index, dh, patch_level_cells[patch_n]);
    }

  protected:
    SmartPointer<const Triangulation<dim, spacedim>> tria;

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;

    // Compressed representation of cells, indexed by patch and then level
    // number. The entries in the IndexSet are the cell indices.
    std::vector<std::vector<IndexSet>> patch_level_cells;
  };
} // namespace fdl

namespace fdl
{
  // ---------------------------- inline functions -----------------------------

  template <int dim, int spacedim>
  const Triangulation<dim, spacedim> &
  PatchMap<dim, spacedim>::get_triangulation() const
  {
    return *tria;
  }



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
    unsigned int cell_level        = level_cells->size(); // max level number
    unsigned int cell_index        = numbers::invalid_unsigned_int;
    unsigned int cummulative_cells = 0;
    Assert(0 <= index, ExcMessage("invalid iterator"));
    for (unsigned int level_n = 0; level_n < level_cells->size(); ++level_n)
      {
        if (index < cummulative_cells + (*level_cells)[level_n].n_elements())
          {
            cell_level = level_n;
            Assert(index >= cummulative_cells, ExcFDLInternalError());
            cell_index = (*level_cells)[level_n].nth_index_in_set(
              index - cummulative_cells);
            break;
          }
        else
          {
            cummulative_cells += (*level_cells)[level_n].n_elements();
          }
      }
    if (cell_level == level_cells->size() && index == cummulative_cells)
      return dh->end();
    else if (cell_level == level_cells->size())
      Assert(index < cummulative_cells, ExcMessage("invalid iterator"));

    // The index has to be equal to or larger than the indices of all cells on
    // coarser levels
    return typename DoFHandler<dim, spacedim>::active_cell_iterator(
      &dh->get_triangulation(), cell_level, cell_index, dh);
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
    Assert(other.dh == this->dh && other.level_cells == this->level_cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return index - other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator<(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.level_cells == this->level_cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index < other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator==(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.level_cells == this->level_cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index == other.index;
  }



  template <int dim, int spacedim>
  bool
  PatchMap<dim, spacedim>::iterator::operator!=(const iterator &other) const
  {
    Assert(other.dh == this->dh && other.level_cells == this->level_cells,
           ExcMessage(
             "only iterators pointing to the same container can be compared."));
    return this->index != other.index;
  }
} // namespace fdl

#endif
