#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/interaction_utilities.h>

#include <fiddle/transfer/overlap_partitioning_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/rtree.h>

#include <boost/container/small_vector.hpp>

#include <ibtk/IndexUtilities.h>
#include <ibtk/LEInteractor.h>

#include <memory>
#include <type_traits>
#include <vector>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  namespace
  {
    // There is no clean way to check this since we treat side-centered
    // data in a different way
    template <int dim, int spacedim, typename patch_type>
    void
    check_depth(const tbox::Pointer<patch_type> &   f_data,
                const FiniteElement<dim, spacedim> &fe)
    {
      const int depth = f_data->getDepth();
      if (std::is_same<patch_type, pdat::SideData<spacedim, double>>::value)
        {
          Assert(depth * spacedim == fe.n_components(),
                 ExcMessage("The depth of the SAMRAI variable should equal the "
                            "number of components of the finite element."));
        }
      else
        {
          Assert(depth == fe.n_components(),
                 ExcMessage("The depth of the SAMRAI variable should equal the "
                            "number of components of the finite element."));
        }
    }

    template <int dim, int spacedim>
    void
    check_quadratures(const std::vector<unsigned char> &  quadrature_indices,
                      const std::vector<Quadrature<dim>> &quadratures,
                      const Triangulation<dim, spacedim> &tria)
    {
      Assert(quadrature_indices.size() == tria.n_active_cells(),
             ExcMessage(
               "There should be exactly one quadrature rule per active cell."));
      if (quadrature_indices.size() > 0)
        {
          Assert(*std::max_element(quadrature_indices.begin(),
                                   quadrature_indices.end()) <
                   quadratures.size(),
                 ExcMessage("Not enough quadrature rules"));
        }
    }
  } // namespace


  template <int spacedim, typename Number, typename TYPE>
  void
  tag_cells_internal(
    const std::vector<BoundingBox<spacedim, Number>> &        bboxes,
    const int                                                 tag_index,
    SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<spacedim>> patch_level)
  {
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

            // We just need an iterator to get a valid patch (so we can get a
            // valid type), so we are done at this point
            break;
          }
      }
  }



  template <int dim, int spacedim, typename TYPE>
  void
  count_quadrature_points_internal(
    const int                           qp_data_idx,
    PatchMap<dim, spacedim> &           patch_map,
    const Mapping<dim, spacedim> &      X_mapping,
    const std::vector<unsigned char> &  quadrature_indices,
    const std::vector<Quadrature<dim>> &quadratures)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      patch_map.get_triangulation());

    // We probably don't need more than 16 quadrature rules
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_X_fe_values;
    // PatchMap only supports looping over DoFHandler iterators, so we need to
    // make one and never use it explicitly
    const Triangulation<dim, spacedim> &tria = patch_map.get_triangulation();
    // No mixed meshes yet
    Assert(tria.get_reference_cells().size() == 1, ExcNotImplemented());
    const ReferenceCell reference_cell = tria.get_reference_cells().front();
    FE_Nothing<dim, spacedim> fe_nothing(reference_cell);
    DoFHandler<dim, spacedim> dof_handler(tria);
    dof_handler.distribute_dofs(fe_nothing);
    for (const Quadrature<dim> &quad : quadratures)
      {
        all_X_fe_values.emplace_back(std::make_unique<FEValues<dim, spacedim>>(
          X_mapping, fe_nothing, quad, update_quadrature_points));
      }

    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto patch = patch_map.get_patch(patch_n);
        tbox::Pointer<pdat::CellData<spacedim, TYPE>> qp_data =
          patch->getPatchData(qp_data_idx);
        Assert(qp_data, ExcMessage("Type mismatch"));
        Assert(qp_data->getDepth() == 1, ExcMessage("depth should be 1"));
        const hier::Box<spacedim> &patch_box = patch->getBox();
        tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> patch_geom =
          patch->getPatchGeometry();
        Assert(patch_geom, ExcMessage("Type mismatch"));

        auto       iter = patch_map.begin(patch_n, dof_handler);
        const auto end  = patch_map.end(patch_n, dof_handler);
        for (; iter != end; ++iter)
          {
            const auto cell = *iter;
            const auto quad_index =
              quadrature_indices[cell->active_cell_index()];

            FEValues<dim, spacedim> &X_fe_values = *all_X_fe_values[quad_index];
            X_fe_values.reinit(cell);
            for (const Point<spacedim> &q_point :
                 X_fe_values.get_quadrature_points())
              {
                const hier::Index<spacedim> i =
                  IBTK::IndexUtilities::getCellIndex(q_point,
                                                     patch_geom,
                                                     patch_box);
                if (patch_box.contains(i))
                  (*qp_data)(i) += TYPE(1);
              }
          }
      }
  }



  template <int dim, int spacedim>
  void
  count_quadrature_points(const int                         qp_data_idx,
                          PatchMap<dim, spacedim> &         patch_map,
                          const Mapping<dim, spacedim> &    X_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<dim>> &quadratures)
  {
    // SAMRAI doesn't offer a way to dispatch on data type so we have to do it
    // ourselves
    if (patch_map.size() == 0)
      {
        return;
      }
    else
      {
        const tbox::Pointer<hier::Patch<spacedim>> patch =
          patch_map.get_patch(0);

        const tbox::Pointer<pdat::CellData<spacedim, int>> int_data =
          patch->getPatchData(qp_data_idx);
        const tbox::Pointer<pdat::CellData<spacedim, float>> float_data =
          patch->getPatchData(qp_data_idx);
        const tbox::Pointer<pdat::CellData<spacedim, double>> double_data =
          patch->getPatchData(qp_data_idx);

        if (int_data)
          count_quadrature_points_internal<dim, spacedim, int>(
            qp_data_idx, patch_map, X_mapping, quadrature_indices, quadratures);
        else if (float_data)
          count_quadrature_points_internal<dim, spacedim, float>(
            qp_data_idx, patch_map, X_mapping, quadrature_indices, quadratures);
        else if (double_data)
          count_quadrature_points_internal<dim, spacedim, double>(
            qp_data_idx, patch_map, X_mapping, quadrature_indices, quadratures);
        else
          Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, int spacedim, typename patch_type>
  void
  compute_projection_rhs_internal(
    const int                           f_data_idx,
    const PatchMap<dim, spacedim> &     patch_map,
    const Mapping<dim, spacedim> &      X_mapping,
    const std::vector<unsigned char> &  quadrature_indices,
    const std::vector<Quadrature<dim>> &quadratures,
    const DoFHandler<dim, spacedim> &   F_dof_handler,
    const Mapping<dim, spacedim> &      F_mapping,
    Vector<double> &                    F_rhs)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      F_dof_handler.get_triangulation());
    const FiniteElement<dim, spacedim> &f_fe          = F_dof_handler.get_fe();
    const unsigned int                  dofs_per_cell = f_fe.dofs_per_cell;
    // TODO - do we need to assume something about the block structure of the
    // FE?

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
          F_mapping, f_fe, quad, update_JxW_values | update_values));
      }

    Vector<double>      cell_rhs(dofs_per_cell);
    std::vector<double> F_values;

    std::vector<types::global_dof_index> dof_indices(f_fe.dofs_per_cell);
    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto                      patch  = patch_map.get_patch(patch_n);
        tbox::Pointer<patch_type> f_data = patch->getPatchData(f_data_idx);
        check_depth(f_data, f_fe);

        auto       iter = patch_map.begin(patch_n, F_dof_handler);
        const auto end  = patch_map.end(patch_n, F_dof_handler);
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
            F_values.resize(f_fe.n_components() * n_q_points);


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
                                            f_fe.n_components(),
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
                if (f_fe.n_components() == 1)
                  {
                    AssertIndexRange(qp_n, F_values.size());
                    const double F_qp = F_values[qp_n];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        cell_rhs[i] += F_fe_values.shape_value(i, qp_n) * F_qp *
                                       F_fe_values.JxW(qp_n);
                      }
                  }
                else if (f_fe.n_components() == spacedim)
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

            F_rhs.add(dof_indices, cell_rhs);
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
                         const DoFHandler<dim, spacedim> &   F_dof_handler,
                         const Mapping<dim, spacedim> &      F_mapping,
                         Vector<double> &                    F_rhs)
  {
#define ARGUMENTS                                                    \
  f_data_idx, patch_map, X_mapping, quadrature_indices, quadratures, \
    F_dof_handler, F_mapping, F_rhs
    if (patch_map.size() != 0)
      {
        auto patch_data = patch_map.get_patch(0)->getPatchData(f_data_idx);
        auto pair       = extract_types(patch_data);

        AssertThrow(pair.second == SAMRAIFieldType::Double,
                    ExcNotImplemented());
        switch (pair.first)
          {
            case SAMRAIPatchType::Edge:
              compute_projection_rhs_internal<dim,
                                              spacedim,
                                              pdat::EdgeData<spacedim, double>>(
                ARGUMENTS);
              break;

            case SAMRAIPatchType::Cell:
              compute_projection_rhs_internal<dim,
                                              spacedim,
                                              pdat::CellData<spacedim, double>>(
                ARGUMENTS);
              break;

            case SAMRAIPatchType::Side:
              compute_projection_rhs_internal<dim,
                                              spacedim,
                                              pdat::SideData<spacedim, double>>(
                ARGUMENTS);
              break;

            case SAMRAIPatchType::Node:
              compute_projection_rhs_internal<dim,
                                              spacedim,
                                              pdat::NodeData<spacedim, double>>(
                ARGUMENTS);
              break;
          }
      }
#undef ARGUMENTS
  }


  // Generic function for retrieving scalar or vector-valued finite element
  // fields
  template <int dim, int spacedim, typename value_type>
  void
  compute_values_generic(const FEValues<dim, spacedim> &fe_values,
                         const Vector<double>           fe_solution,
                         std::vector<value_type> &      F_values)
  {
    Assert(false, ExcNotImplemented());
  }

  template <int dim, int spacedim>
  void
  compute_values_generic(const FEValues<dim, spacedim> &fe_values,
                         const Vector<double>           fe_solution,
                         std::vector<double> &          F_values)
  {
    fe_values.get_function_values(fe_solution, F_values);
  }

  template <int dim, int spacedim>
  void
  compute_values_generic(const FEValues<dim, spacedim> &   fe_values,
                         const Vector<double>              fe_solution,
                         std::vector<Tensor<1, spacedim>> &F_values)
  {
    const FEValuesExtractors::Vector vec(0);
    fe_values[vec].get_function_values(fe_solution, F_values);
  }

  template <int dim, int spacedim, typename value_type, typename patch_type>
  void
  compute_spread_internal(const int                         f_data_idx,
                          PatchMap<dim, spacedim> &         patch_map,
                          const Mapping<dim, spacedim> &    X_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<dim>> &quadratures,
                          const DoFHandler<dim, spacedim> &   F_dof_handler,
                          const Mapping<dim, spacedim> &      F_mapping,
                          const Vector<double> &              F)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      F_dof_handler.get_triangulation());
    const FiniteElement<dim, spacedim> &f_fe          = F_dof_handler.get_fe();
    const unsigned int                  dofs_per_cell = f_fe.dofs_per_cell;

    // We probably don't need more than 16 quadrature rules
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_X_fe_values;
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_F_fe_values;
    for (const Quadrature<dim> &quad : quadratures)
      {
        all_X_fe_values.emplace_back(std::make_unique<FEValues<dim, spacedim>>(
          X_mapping, f_fe, quad, update_quadrature_points));
        all_F_fe_values.emplace_back(std::make_unique<FEValues<dim, spacedim>>(
          F_mapping, f_fe, quad, update_JxW_values | update_values));
      }

    std::vector<value_type> F_values;

    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto                      patch  = patch_map.get_patch(patch_n);
        tbox::Pointer<patch_type> f_data = patch->getPatchData(f_data_idx);
        Assert(f_data, ExcMessage("Type mismatch"));
        check_depth(f_data, f_fe);

        auto       iter = patch_map.begin(patch_n, F_dof_handler);
        const auto end  = patch_map.end(patch_n, F_dof_handler);
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
            Assert(F_fe_values.get_quadrature() == X_fe_values.get_quadrature(),
                   ExcFDLInternalError());

            const std::vector<Point<spacedim>> &q_points =
              X_fe_values.get_quadrature_points();
            const unsigned int n_q_points = q_points.size();
            F_values.resize(n_q_points);

            // get forces:
            std::fill(F_values.begin(), F_values.end(), value_type());
            compute_values_generic(F_fe_values, F, F_values);
            for (unsigned int qp = 0; qp < n_q_points; ++qp)
              F_values[qp] *= F_fe_values.JxW(qp);

            // TODO reimplement zeroExteriorValues here

            // spread at quadrature points:
            static_assert(sizeof(Point<spacedim>) == sizeof(double) * spacedim,
                          "FORTRAN routines assume we are packed");
            const auto X_data =
              reinterpret_cast<const double *>(q_points.data());
            // the number of components is determined at run time so use a
            // normal assertion
            AssertThrow(sizeof(value_type) ==
                          sizeof(double) * f_fe.n_components(),
                        ExcMessage("FORTRAN routines assume we are packed"));
            const auto F_data =
              reinterpret_cast<const double *>(F_values.data());

            IBTK::LEInteractor::spread(f_data,
                                       F_data,
                                       F_values.size() * f_fe.n_components(),
                                       f_fe.n_components(),
                                       X_data,
                                       n_q_points * spacedim,
                                       spacedim,
                                       patch,
                                       patch->getBox(),
                                       "BSPLINE_3");
          }
      }
  }



  template <int dim, int spacedim>
  void
  compute_spread(const int                           f_data_idx,
                 PatchMap<dim, spacedim> &           patch_map,
                 const Mapping<dim, spacedim> &      X_mapping,
                 const std::vector<unsigned char> &  quadrature_indices,
                 const std::vector<Quadrature<dim>> &quadratures,
                 const DoFHandler<dim, spacedim> &   F_dof_handler,
                 const Mapping<dim, spacedim> &      F_mapping,
                 const Vector<double> &              F)
  {
#define ARGUMENTS                                                    \
  f_data_idx, patch_map, X_mapping, quadrature_indices, quadratures, \
    F_dof_handler, F_mapping, F
    if (patch_map.size() != 0)
      {
        auto patch_data = patch_map.get_patch(0)->getPatchData(f_data_idx);
        auto pair       = extract_types(patch_data);

        AssertThrow(pair.second == SAMRAIFieldType::Double,
                    ExcNotImplemented());
        switch (pair.first)
          {
            case SAMRAIPatchType::Edge:
              switch (extract_depth(patch_data))
                {
                  case 1:
                    compute_spread_internal<dim,
                                            spacedim,
                                            double,
                                            pdat::EdgeData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  case spacedim:
                    compute_spread_internal<dim,
                                            spacedim,
                                            Tensor<1, spacedim>,
                                            pdat::EdgeData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  default:
                    AssertThrow(false, ExcNotImplemented());
                }
              break;

            case SAMRAIPatchType::Cell:
              switch (extract_depth(patch_data))
                {
                  case 1:
                    compute_spread_internal<dim,
                                            spacedim,
                                            double,
                                            pdat::CellData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  case spacedim:
                    compute_spread_internal<dim,
                                            spacedim,
                                            Tensor<1, spacedim>,
                                            pdat::CellData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  default:
                    AssertThrow(false, ExcNotImplemented());
                }
              break;

            case SAMRAIPatchType::Side:
              // We only support depth == 1 for side-centered
              Assert(extract_depth(patch_data) == 1, ExcFDLNotImplemented());
              compute_spread_internal<dim,
                                      spacedim,
                                      Tensor<1, spacedim>,
                                      pdat::SideData<spacedim, double>>(
                ARGUMENTS);
              break;

            case SAMRAIPatchType::Node:
              switch (extract_depth(patch_data))
                {
                  case 1:
                    compute_spread_internal<dim,
                                            spacedim,
                                            double,
                                            pdat::NodeData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  case spacedim:
                    compute_spread_internal<dim,
                                            spacedim,
                                            Tensor<1, spacedim>,
                                            pdat::NodeData<spacedim, double>>(
                      ARGUMENTS);
                    break;
                  default:
                    AssertThrow(false, ExcNotImplemented());
                }
              break;
          }
      }
#undef ARGUMENTS
  }

  // instantiations

  template void
  tag_cells(const std::vector<BoundingBox<NDIM, float>> &         bboxes,
            const int                                             tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);

  template void
  tag_cells(const std::vector<BoundingBox<NDIM, double>> &        bboxes,
            const int                                             tag_index,
            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level);

  template void
  count_quadrature_points(const int                         qp_data_idx,
                          PatchMap<NDIM - 1, NDIM> &        patch_map,
                          const Mapping<NDIM - 1, NDIM> &   X_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<NDIM - 1>> &quadratures);

  template void
  count_quadrature_points(const int                         qp_data_idx,
                          PatchMap<NDIM, NDIM> &            patch_map,
                          const Mapping<NDIM, NDIM> &       X_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<NDIM>> &quadratures);

  template void
  compute_projection_rhs(const int                         f_data_idx,
                         const PatchMap<NDIM - 1, NDIM> &  patch_map,
                         const Mapping<NDIM - 1, NDIM> &   X_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM - 1>> &quadratures,
                         const DoFHandler<NDIM - 1, NDIM> &       F_dof_handler,
                         const Mapping<NDIM - 1, NDIM> &          F_mapping,
                         Vector<double> &                         F_rhs);

  template void
  compute_projection_rhs(const int                         f_data_idx,
                         const PatchMap<NDIM> &            patch_map,
                         const Mapping<NDIM> &             X_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM>> &quadratures,
                         const DoFHandler<NDIM> &             F_dof_handler,
                         const Mapping<NDIM> &                F_mapping,
                         Vector<double> &                     F_rhs);

  template void
  compute_spread(const int                                f_data_idx,
                 PatchMap<NDIM - 1, NDIM> &               patch_map,
                 const Mapping<NDIM - 1, NDIM> &          X_mapping,
                 const std::vector<unsigned char> &       quadrature_indices,
                 const std::vector<Quadrature<NDIM - 1>> &quadratures,
                 const DoFHandler<NDIM - 1, NDIM> &       F_dof_handler,
                 const Mapping<NDIM - 1, NDIM> &          F_mapping,
                 const Vector<double> &                   F);

  template void
  compute_spread(const int                            f_data_idx,
                 PatchMap<NDIM, NDIM> &               patch_map,
                 const Mapping<NDIM, NDIM> &          X_mapping,
                 const std::vector<unsigned char> &   quadrature_indices,
                 const std::vector<Quadrature<NDIM>> &quadratures,
                 const DoFHandler<NDIM, NDIM> &       F_dof_handler,
                 const Mapping<NDIM, NDIM> &          F_mapping,
                 const Vector<double> &               F);
} // namespace fdl
