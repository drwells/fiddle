#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/interaction.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/rtree.h>

#include <boost/container/small_vector.hpp>

#include <ibtk/IndexUtilities.h>
#include <ibtk/LEInteractor.h>

#include <memory>
#include <vector>

namespace fdl
{
  using namespace dealii;

  template <int spacedim, typename Number, typename TYPE>
  void
  tag_cells_internal(
    const std::vector<BoundingBox<spacedim, Number>> &        bboxes,
    const int                                                 tag_index,
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
    using namespace SAMRAI;

    const tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geom =
      patch_level->getGridGeometry();
    const hier::IntVector<spacedim> ratio = patch_level->getRatio();

    const std::vector<tbox::Pointer<hier::Patch<spacedim>>> patches =
      extract_patches(patch_level);

    std::vector<tbox::Pointer<pdat::CellData<spacedim, TYPE>>> tag_data;
    for (const auto &patch : patches)
      {
        Assert(patch->getPatchData(tag_index),
               ExcMessage("should be a pointer here"));
        tag_data.push_back(patch->getPatchData(tag_index));
      }
    const std::vector<BoundingBox<spacedim, float>> patch_bboxes =
      compute_patch_bboxes<spacedim, float>(patches);
    const auto rtree = pack_rtree_of_indices(patch_bboxes);

    // loop over element bboxes...
    for (const auto &bbox : bboxes)
      {
        const hier::Index<spacedim> i_lower =
          IBTK::IndexUtilities::getCellIndex(bbox.get_boundary_points().first,
                                             grid_geom,
                                             ratio);
        const hier::Index<spacedim> i_upper =
          IBTK::IndexUtilities::getCellIndex(bbox.get_boundary_points().second,
                                             grid_geom,
                                             ratio);
        const hier::Box<spacedim> box(i_lower, i_upper);

        // and determine which patches each intersects.
        namespace bgi = boost::geometry::index;
        // TODO: this still allocates memory. We should use something else to
        // avoid that, like boost::function_to_output_iterator or our own
        // equivalent
        for (const std::size_t patch_n :
             rtree | bgi::adaptors::queried(bgi::intersects(bbox)))
          {
            AssertIndexRange(patch_n, patches.size());
            tag_data[patch_n]->fillAll(TYPE(1), box);
          }
      }
  }

  /**
   * Tag cells in the patch hierarchy that intersect the provided bounding
   * boxes.
   */
  template <int spacedim, typename Number>
  void
  tag_cells(
    const std::vector<BoundingBox<spacedim, Number>> &        bboxes,
    const int                                                 tag_index,
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
    // SAMRAI doesn't offer a way to dispatch on data type so we have to do it
    // ourselves
    using namespace SAMRAI;

    if (patch_level->getNumberOfPatches() == 0)
      {
        return;
      }
    else
      {
        for (typename hier::PatchLevel<spacedim>::Iterator p(patch_level); p;
             p++)
          {
            const tbox::Pointer<hier::Patch<spacedim>> patch =
              patch_level->getPatch(p());

            const tbox::Pointer<pdat::CellData<spacedim, int>> int_data =
              patch->getPatchData(tag_index);
            const tbox::Pointer<pdat::CellData<spacedim, float>> float_data =
              patch->getPatchData(tag_index);
            const tbox::Pointer<pdat::CellData<spacedim, double>> double_data =
              patch->getPatchData(tag_index);

            if (int_data)
              tag_cells_internal<spacedim, Number, int>(bboxes,
                                                        tag_index,
                                                        patch_level);
            else if (float_data)
              tag_cells_internal<spacedim, Number, float>(bboxes,
                                                          tag_index,
                                                          patch_level);
            else if (double_data)
              tag_cells_internal<spacedim, Number, double>(bboxes,
                                                           tag_index,
                                                           patch_level);
            else
              Assert(false, ExcNotImplemented());

            break;
          }
      }
  }



  template <int dim, int spacedim>
  void
  compute_projection_rhs(const int                           f_data_idx,
                         const PatchMap<dim, spacedim> &     patch_map,
                         const Mapping<dim, spacedim> &      X_mapping,
                         const std::vector<unsigned char> &  quadrature_indices,
                         const std::vector<Quadrature<dim>> &quadratures,
                         const DoFHandler<dim, spacedim> &   f_dof_handler,
                         const Mapping<dim, spacedim> &      f_mapping,
                         Vector<double> &                    f_rhs)
  {
    using namespace SAMRAI;

    Assert(quadrature_indices.size() ==
             f_dof_handler.get_triangulation().n_active_cells(),
           ExcMessage(
             "There should be exactly one quadrature rule per active cell."));
    if (quadrature_indices.size() > 0)
      {
        Assert(*std::max_element(quadrature_indices.begin(),
                                 quadrature_indices.end()) < quadratures.size(),
               ExcMessage("Not enough quadrature rules"));
      }

    const FiniteElement<dim, spacedim> &f_fe          = f_dof_handler.get_fe();
    const unsigned int                  dofs_per_cell = f_fe.dofs_per_cell;
    Assert(f_fe.n_blocks() == 1, ExcNotImplemented());

    // We probably don't need more than 16 quadrature rules
    //
    // TODO - implement a move constructor for FEValues
    // The FE for X is arbitrary - we just need quadrature points. The actual X
    // FE is in X_mapping
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_X_fe_values;
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_F_fe_values;
    for (const Quadrature<dim> &quad : quadratures)
      {
        all_X_fe_values.emplace_back(std::make_unique<FEValues<dim, spacedim>>(
          X_mapping, f_fe, quad, update_quadrature_points));
        all_F_fe_values.emplace_back(std::make_unique<FEValues<dim, spacedim>>(
          f_mapping, f_fe, quad, update_JxW_values | update_values));
      }

    Vector<double>      cell_rhs(dofs_per_cell);
    std::vector<double> F_values;

    std::vector<types::global_dof_index> dof_indices(f_fe.dofs_per_cell);
    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto patch = patch_map.get_patch(patch_n);
        tbox::Pointer<pdat::CellData<spacedim, double>> f_data =
          patch->getPatchData(f_data_idx);
        // Assert(f_data, ExcMessage("Only side-centered data is supported
        // ATM"));
        Assert(f_data->getDepth() == f_fe.n_components(),
               ExcMessage("The depth of the SAMRAI variable should equal the "
                          "number of components of the finite element."));
        const unsigned int depth = f_data->getDepth();

        auto       iter = patch_map.begin(patch_n, f_dof_handler);
        const auto end  = patch_map.end(patch_n, f_dof_handler);
        for (; iter != end; ++iter)
          {
            const auto cell = *iter;
            const auto quad_index =
              quadrature_indices[cell->active_cell_index()];

            // Reinitialize:
            FEValues<dim, spacedim> &F_fe_values = *all_F_fe_values[quad_index];
            FEValues<dim, spacedim> &X_fe_values = *all_X_fe_values[quad_index];
            F_fe_values.reinit(cell);
            X_fe_values.reinit(cell);
            F_values.resize(X_fe_values.get_quadrature_points().size());
            Assert(F_fe_values.get_quadrature() == X_fe_values.get_quadrature(),
                   ExcFDLInternalError());

            cell_rhs = 0.0;
            cell->get_dof_indices(dof_indices);
            const std::vector<Point<spacedim>> &q_points =
              X_fe_values.get_quadrature_points();
            const unsigned int n_q_points = q_points.size();
            F_values.resize(depth * n_q_points);


            // Interpolate at quadrature points:
#if 1
            // Interpolate values from the patch. This LEInteractor call could
            // be improved - we don't need to look up certain things on each
            // element
            static_assert(sizeof(Point<spacedim>) == sizeof(double) * spacedim,
                          "FORTRAN routines assume we are packed");
            const auto X_data =
              reinterpret_cast<const double *>(q_points.data());
            std::fill(F_values.begin(), F_values.end(), 0.0);
            IBTK::LEInteractor::interpolate(F_values.data(),
                                            F_values.size(),
                                            depth,
                                            X_data,
                                            q_points.size() * spacedim,
                                            spacedim,
                                            f_data,
                                            patch,
                                            patch->getBox(),
                                            "BSPLINE_3");
#else
            std::fill(F_values.begin(), F_values.end(), 1.0);
#endif

            for (unsigned int qp_n = 0; qp_n < n_q_points; ++qp_n)
              {
                if (depth == 1)
                  {
                    AssertIndexRange(qp_n, F_values.size());
                    const double F_qp = F_values[qp_n];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        cell_rhs[i] += F_fe_values.shape_value(i, qp_n) * F_qp *
                                       F_fe_values.JxW(qp_n);
                      }
                  }
                else if (depth == spacedim)
                  {
                    Tensor<1, spacedim> F_qp;
                    for (unsigned int d = 0; d < spacedim; ++d)
                      {
                        AssertIndexRange(qp_n * spacedim + d, F_values.size());
                        F_qp[d] = F_values[qp_n * spacedim + d];
                      }

                    // TODO - this only works with primitive elements
                    // TODO - perhaps its worth unrolling this loop?
                    // dofs_per_cell is probably 12, 30, or 45.
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const unsigned int component =
                          f_fe.system_to_component_index(i).first;
                        cell_rhs[i] += F_fe_values.shape_value(i, qp_n) *
                                       F_qp[component] * F_fe_values.JxW(qp_n);
                      }
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }
              }

            f_rhs.add(dof_indices, cell_rhs);
          }
      }
  }

  template void
  tag_cells(const std::vector<BoundingBox<NDIM, float>> &         bboxes,
            const int                                             tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);

  template void
  tag_cells(const std::vector<BoundingBox<NDIM, double>> &        bboxes,
            const int                                             tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);

  template void
  compute_projection_rhs(const int                         f_data_idx,
                         const PatchMap<NDIM - 1, NDIM> &  patch_map,
                         const Mapping<NDIM - 1, NDIM> &   X_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM - 1>> &quadratures,
                         const DoFHandler<NDIM - 1, NDIM> &       f_dof_handler,
                         const Mapping<NDIM - 1, NDIM> &          f_mapping,
                         Vector<double> &                         f_rhs);

  template void
  compute_projection_rhs(const int                         f_data_idx,
                         const PatchMap<NDIM> &            patch_map,
                         const Mapping<NDIM> &             X_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM>> &quadratures,
                         const DoFHandler<NDIM> &             f_dof_handler,
                         const Mapping<NDIM> &                f_mapping,
                         Vector<double> &                     f_rhs);

} // namespace fdl