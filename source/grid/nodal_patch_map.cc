#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/nodal_patch_map.h>

#include <deal.II/numerics/rtree.h>

#include <algorithm>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim>
  NodalPatchMap<dim, spacedim>::NodalPatchMap(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const std::vector<std::vector<BoundingBox<spacedim>>> &patch_bboxes,
    const Vector<double> &nodal_coordinates)
  {
    reinit(patches, patch_bboxes, nodal_coordinates);
  }



  template <int dim, int spacedim>
  void
  NodalPatchMap<dim, spacedim>::reinit(
    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> &patches,
    const std::vector<std::vector<BoundingBox<spacedim>>> &patch_bboxes,
    const Vector<double> &nodal_coordinates)
  {
    AssertDimension(patches.size(), patch_bboxes.size());
    if (patches.size() == 0)
      return;

    Assert(nodal_coordinates.size() % spacedim == 0,
           ExcMessage(
             "There should be N * spacedim entries in nodal_coordinates"));
    const std::size_t n_nodes = nodal_coordinates.size() / spacedim;

    this->patches = patches;
    patch_dof_indices.resize(0);
    for (std::size_t i = 0; i < patches.size(); ++i)
      patch_dof_indices.emplace_back(
        static_cast<types::global_dof_index>(nodal_coordinates.size()));

    std::vector<int> patch_levels;
    for (auto &patch : patches)
      patch_levels.push_back(patch->getPatchLevelNumber());

    const int min_ln =
      *std::min_element(patch_levels.begin(), patch_levels.end());
    const int max_ln =
      *std::max_element(patch_levels.begin(), patch_levels.end());

    for (int ln = max_ln; ln >= min_ln; --ln)
      {
        std::vector<std::size_t>           level_patch_indices;
        std::vector<BoundingBox<spacedim>> level_patch_bboxes;
        for (std::size_t i = 0; i < patches.size(); ++i)
          if (patch_levels[i] == ln)
            for (const auto &bbox : patch_bboxes[i])
              {
                level_patch_indices.push_back(i);
                level_patch_bboxes.push_back(bbox);
              }

        // no need to continue if there are no locally owned boxes on this level
        if (level_patch_bboxes.size() == 0)
          continue;
        const auto rtree = pack_rtree_of_indices(level_patch_bboxes);
        for (std::size_t node_n = 0; node_n < n_nodes; ++node_n)
          {
            Point<spacedim> node;
            for (unsigned int d = 0; d < spacedim; ++d)
              node[d] = nodal_coordinates[spacedim * node_n + d];

            namespace bgi = boost::geometry::index;
            for (const std::size_t level_patch_n :
                 rtree | bgi::adaptors::queried(bgi::intersects(node)))
              {
                AssertIndexRange(level_patch_n, level_patch_indices.size());
                const auto patch_n = level_patch_indices[level_patch_n];
                patch_dof_indices[patch_n].add_range(spacedim * node_n,
                                                     spacedim * node_n +
                                                       spacedim);
              }
          }
      }

    for (IndexSet &index_set : patch_dof_indices)
      index_set.compress();
  }



  template class NodalPatchMap<NDIM - 1, NDIM>;
  template class NodalPatchMap<NDIM, NDIM>;
} // namespace fdl
