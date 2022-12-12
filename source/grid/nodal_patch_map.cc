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

    const std::vector<BoundingBox<spacedim, double>> patch_bboxes =
      compute_patch_bboxes<spacedim, double>(patches,
                                             extra_ghost_cell_fraction);
    std::vector<int> patch_levels;
    for (auto &patch : patches)
      patch_levels.push_back(patch->getPatchLevelNumber());

    const int min_ln =
      *std::min_element(patch_levels.begin(), patch_levels.end());
    const int max_ln =
      *std::max_element(patch_levels.begin(), patch_levels.end());

    // permit nodes to be assigned to multiple patches on the same level (i.e.,
    // they may be in a ghost region of one patch but the interior of another)
    // but do not permit them to be assigned to multiple levels at once.
    //
    // TODO: a more careful implementation could remove that restriction by
    // replacing the coarser level bounding boxes with a new set of bounding
    // boxes which do not overlap with the finer level.
    std::vector<bool> node_assigned_to_finer_level(n_nodes);
    for (int ln = max_ln; ln >= min_ln; --ln)
      {
        std::vector<bool> node_assigned_to_current_level(n_nodes);
        // Speed up intersection by putting the patch bboxes in an rtree
        std::vector<std::size_t>                   level_patch_indices;
        std::vector<BoundingBox<spacedim, double>> level_patch_bboxes;
        for (std::size_t i = 0; i < patch_bboxes.size(); ++i)
          if (patch_levels[i] == ln)
            {
              level_patch_indices.push_back(i);
              level_patch_bboxes.push_back(patch_bboxes[i]);
            }

        const auto rtree = pack_rtree_of_indices(level_patch_bboxes);

        for (std::size_t node_n = 0; node_n < n_nodes; ++node_n)
          {
            if (node_assigned_to_finer_level[node_n])
              continue;

            Point<spacedim> node;
            for (unsigned int d = 0; d < spacedim; ++d)
              node[d] = nodal_coordinates[spacedim * node_n + d];

            namespace bgi = boost::geometry::index;
            for (const std::size_t level_patch_n :
                 rtree | bgi::adaptors::queried(bgi::intersects(node)))
              {
                AssertIndexRange(level_patch_n, level_patch_indices.size());
                std::size_t patch_n = level_patch_indices[level_patch_n];

                patch_dof_indices[patch_n].add_range(spacedim * node_n,
                                                     spacedim * node_n +
                                                       spacedim);
                node_assigned_to_current_level[node_n] = true;
              }
          }

        for (std::size_t i = 0; i < n_nodes; ++i)
          node_assigned_to_finer_level[i] = node_assigned_to_finer_level[i] ||
                                            node_assigned_to_current_level[i];
      }

    for (IndexSet &index_set : patch_dof_indices)
      index_set.compress();
  }


  template class NodalPatchMap<NDIM - 1, NDIM>;
  template class NodalPatchMap<NDIM, NDIM>;
} // namespace fdl
