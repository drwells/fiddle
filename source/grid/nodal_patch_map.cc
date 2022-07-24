#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/nodal_patch_map.h>

#include <deal.II/numerics/rtree.h>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim>
  NodalPatchMap<dim, spacedim>::NodalPatchMap(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const double          extra_ghost_cell_fraction,
    const Vector<double> &nodal_coordinates)
  {
    reinit(patches, extra_ghost_cell_fraction, nodal_coordinates);
  }

  template <int dim, int spacedim>
  void
  NodalPatchMap<dim, spacedim>::reinit(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const double          extra_ghost_cell_fraction,
    const Vector<double> &nodal_coordinates)
  {
    Assert(nodal_coordinates.size() % spacedim == 0,
           ExcMessage(
             "There should be N*spacedim entries in nodal_coordinates"));

    this->patches = patches;
    patch_dof_indices.resize(0);
    for (std::size_t i = 0; i < patches.size(); ++i)
      patch_dof_indices.emplace_back(
        static_cast<types::global_dof_index>(nodal_coordinates.size()));

    const std::vector<BoundingBox<spacedim, double>> patch_bboxes =
      compute_patch_bboxes<spacedim, double>(patches,
                                             extra_ghost_cell_fraction);

    // Speed up intersection by putting the patch bboxes in an rtree
    const auto rtree = pack_rtree_of_indices(patch_bboxes);

    for (std::size_t node_n = 0; node_n < nodal_coordinates.size() / spacedim;
         ++node_n)
      {
        Point<spacedim> node;
        for (unsigned int d = 0; d < spacedim; ++d)
          node[d] = nodal_coordinates[spacedim * node_n + d];

        namespace bgi = boost::geometry::index;
        for (const std::size_t patch_n :
             rtree | bgi::adaptors::queried(bgi::intersects(node)))
          {
            AssertIndexRange(patch_n, patches.size());

            patch_dof_indices[patch_n].add_range(spacedim * node_n,
                                                 spacedim * node_n + spacedim);
          }
      }

    for (IndexSet &index_set : patch_dof_indices)
      index_set.compress();
  }


  template class NodalPatchMap<NDIM - 1, NDIM>;
  template class NodalPatchMap<NDIM, NDIM>;
} // namespace fdl
