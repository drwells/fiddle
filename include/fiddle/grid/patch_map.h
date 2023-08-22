#ifndef included_fiddle_grid_patch_map_h
#define included_fiddle_grid_patch_map_h

#include <fiddle/base/config.h>

#include <fiddle/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>

FDL_DISABLE_EXTRA_DIAGNOSTICS
#include <Patch.h>
FDL_ENABLE_EXTRA_DIAGNOSTICS

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
               const std::vector<IndexSet>     &patch_level_cells,
               const std::vector<std::size_t>  &patch_cummulative_n_cells);

      const DoFHandler<dim, spacedim> *dh;

      const std::vector<IndexSet>    *level_cells;
      const std::vector<std::size_t> *cummulative_n_cells;

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
    get_patch(const std::size_t patch_n);

    const tbox::Pointer<hier::Patch<spacedim>> &
    get_patch(const std::size_t patch_n) const;

    iterator
    begin(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const;

    iterator
    end(const std::size_t patch_n, const DoFHandler<dim, spacedim> &dh) const;

  protected:
    SmartPointer<const Triangulation<dim, spacedim>> tria;

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches;

    // Compressed representation of cells, indexed by patch and then level
    // number. The entries in the IndexSet are the cell indices.
    std::vector<std::vector<IndexSet>> patch_level_cells;

    // Cumulative number of cells on each level: e.g., the first entry is the
    // number of cells on level 0, the second is the sum of level 0 and 1, etc.
    // The last entry is the total number of cells. Used by the iterators.
    std::vector<std::vector<std::size_t>> cummulative_n_cells;
  };


  // --------------------------- inline functions --------------------------- //


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
  PatchMap<dim, spacedim>::iterator::iterator(
    const std::ptrdiff_t             index,
    const DoFHandler<dim, spacedim> &dof_handler,
    const std::vector<IndexSet>     &patch_level_cells,
    const std::vector<std::size_t>  &patch_cummulative_n_cells)
    : dh(&dof_handler)
    , level_cells(&patch_level_cells)
    , cummulative_n_cells(&patch_cummulative_n_cells)
    , index(index)
  {}



  template <int dim, int spacedim>
  tbox::Pointer<hier::Patch<spacedim>> &
  PatchMap<dim, spacedim>::get_patch(const std::size_t patch_n)
  {
    AssertIndexRange(patch_n, size());
    return patches[patch_n];
  }



  template <int dim, int spacedim>
  const tbox::Pointer<hier::Patch<spacedim>> &
  PatchMap<dim, spacedim>::get_patch(const std::size_t patch_n) const
  {
    AssertIndexRange(patch_n, size());
    return patches[patch_n];
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::begin(const std::size_t                patch_n,
                                 const DoFHandler<dim, spacedim> &dh) const
  {
    AssertIndexRange(patch_n, size());
    Assert(&dh.get_triangulation() == &*tria,
           ExcMessage("must use same Triangulation"));
    return iterator(0,
                    dh,
                    patch_level_cells[patch_n],
                    cummulative_n_cells[patch_n]);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator
  PatchMap<dim, spacedim>::end(const std::size_t                patch_n,
                               const DoFHandler<dim, spacedim> &dh) const
  {
    AssertIndexRange(patch_n, size());
    Assert(&dh.get_triangulation() == &*tria,
           ExcMessage("must use same Triangulation"));
    return iterator(cummulative_n_cells[patch_n].back(),
                    dh,
                    patch_level_cells[patch_n],
                    cummulative_n_cells[patch_n]);
  }



  template <int dim, int spacedim>
  typename PatchMap<dim, spacedim>::iterator::value_type
  PatchMap<dim, spacedim>::iterator::operator*() const
  {
    Assert(0 <= index, ExcMessage("invalid iterator"));
    const auto it = std::upper_bound(cummulative_n_cells->begin(),
                                     cummulative_n_cells->end(),
                                     index);
    if (it == cummulative_n_cells->end())
      {
        Assert(index <= std::ptrdiff_t(cummulative_n_cells->back()),
               ExcMessage("invalid iterator"));
        return dh->end();
      }
    const unsigned int cell_level = it - cummulative_n_cells->begin();
    const unsigned int cell_index = (*level_cells)[cell_level].nth_index_in_set(
      index - (cell_level == 0 ? 0 : (*cummulative_n_cells)[cell_level - 1]));
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
