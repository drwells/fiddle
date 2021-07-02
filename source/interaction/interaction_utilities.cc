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
    check_depth(const tbox::Pointer<patch_type> &   data,
                const FiniteElement<dim, spacedim> &fe)
    {
      const int depth = data->getDepth();
      if (std::is_same<patch_type, pdat::SideData<spacedim, double>>::value)
        {
          Assert(depth * spacedim == int(fe.n_components()),
                 ExcMessage("The depth of the SAMRAI variable should equal the "
                            "number of components of the finite element."));
        }
      else
        {
          Assert(depth == int(fe.n_components()),
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
    // extract what we need for getCellIndex:
    const hier::IntVector<spacedim> ratio = patch_level->getRatio();
    const tbox::Pointer<geom::CartesianGridGeometry<spacedim>> grid_geom =
      patch_level->getGridGeometry();
    const double *const          dx0 = grid_geom->getDx();
    std::array<double, spacedim> dx;
    for (unsigned int d = 0; d < spacedim; ++d)
      dx[d] = dx0[d] / double(ratio(d));
    const auto domain_box =
      hier::Box<spacedim>::refine(grid_geom->getPhysicalDomain()[0], ratio);

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
        // and determine which patches each intersects.
        namespace bgi = boost::geometry::index;
        // TODO: this still allocates memory. We should use something else to
        // avoid that, like boost::function_to_output_iterator or our own
        // equivalent
        for (const std::size_t patch_n :
             rtree | bgi::adaptors::queried(bgi::intersects(bbox)))
          {
            AssertIndexRange(patch_n, patches.size());

            const hier::Index<spacedim> i_lower =
              IBTK::IndexUtilities::getCellIndex(
                bbox.get_boundary_points().first,
                grid_geom->getXLower(),
                grid_geom->getXUpper(),
                dx.data(),
                domain_box.lower(),
                domain_box.upper());
            const hier::Index<spacedim> i_upper =
              IBTK::IndexUtilities::getCellIndex(
                bbox.get_boundary_points().second,
                grid_geom->getXLower(),
                grid_geom->getXUpper(),
                dx.data(),
                domain_box.lower(),
                domain_box.upper());
            const hier::Box<spacedim> box(i_lower, i_upper);

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
    const Mapping<dim, spacedim> &      position_mapping,
    const std::vector<unsigned char> &  quadrature_indices,
    const std::vector<Quadrature<dim>> &quadratures)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      patch_map.get_triangulation());

    // We probably don't need more than 16 quadrature rules
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_position_fe_values;
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
        all_position_fe_values.emplace_back(
          std::make_unique<FEValues<dim, spacedim>>(
            position_mapping, fe_nothing, quad, update_quadrature_points));
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

            FEValues<dim, spacedim> &position_fe_values =
              *all_position_fe_values[quad_index];
            position_fe_values.reinit(cell);
            for (const Point<spacedim> &q_point :
                 position_fe_values.get_quadrature_points())
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
                          const Mapping<dim, spacedim> &    position_mapping,
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
            qp_data_idx,
            patch_map,
            position_mapping,
            quadrature_indices,
            quadratures);
        else if (float_data)
          count_quadrature_points_internal<dim, spacedim, float>(
            qp_data_idx,
            patch_map,
            position_mapping,
            quadrature_indices,
            quadratures);
        else if (double_data)
          count_quadrature_points_internal<dim, spacedim, double>(
            qp_data_idx,
            patch_map,
            position_mapping,
            quadrature_indices,
            quadratures);
        else
          Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, int spacedim, typename patch_type>
  void
  compute_projection_rhs_internal(
    const std::string &                 kernel_name,
    const int                           data_idx,
    const PatchMap<dim, spacedim> &     patch_map,
    const Mapping<dim, spacedim> &      position_mapping,
    const std::vector<unsigned char> &  quadrature_indices,
    const std::vector<Quadrature<dim>> &quadratures,
    const DoFHandler<dim, spacedim> &   dof_handler,
    const Mapping<dim, spacedim> &      mapping,
    Vector<double> &                    rhs)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      dof_handler.get_triangulation());
    const FiniteElement<dim, spacedim> &fe            = dof_handler.get_fe();
    const unsigned int                  dofs_per_cell = fe.dofs_per_cell;
    // TODO - do we need to assume something about the block structure of the
    // FE?

    // We probably don't need more than 16 quadrature rules
    //
    // TODO - implement a move constructor for FEValues
    // The FE for position is arbitrary - we just need quadrature points. The
    // actual position FE is in position_mapping
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_position_fe_values;
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_rhs_fe_values;
    for (const Quadrature<dim> &quad : quadratures)
      {
        all_position_fe_values.emplace_back(
          std::make_unique<FEValues<dim, spacedim>>(
            position_mapping, fe, quad, update_quadrature_points));
        all_rhs_fe_values.emplace_back(
          std::make_unique<FEValues<dim, spacedim>>(
            mapping, fe, quad, update_JxW_values | update_values));
      }

    Vector<double>      cell_rhs(dofs_per_cell);
    std::vector<double> rhs_values;

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto                      patch      = patch_map.get_patch(patch_n);
        tbox::Pointer<patch_type> patch_data = patch->getPatchData(data_idx);
        check_depth(patch_data, fe);

        auto       iter = patch_map.begin(patch_n, dof_handler);
        const auto end  = patch_map.end(patch_n, dof_handler);
        for (; iter != end; ++iter)
          {
            const auto cell = *iter;
            const auto quad_index =
              quadrature_indices[cell->active_cell_index()];

            // Reinitialize:
            FEValues<dim, spacedim> &rhs_fe_values =
              *all_rhs_fe_values[quad_index];
            FEValues<dim, spacedim> &position_fe_values =
              *all_position_fe_values[quad_index];
            rhs_fe_values.reinit(cell);
            position_fe_values.reinit(cell);
            rhs_values.resize(
              position_fe_values.get_quadrature_points().size());
            Assert(rhs_fe_values.get_quadrature() ==
                     position_fe_values.get_quadrature(),
                   ExcFDLInternalError());

            cell_rhs = 0.0;
            cell->get_dof_indices(dof_indices);
            const std::vector<Point<spacedim>> &q_points =
              position_fe_values.get_quadrature_points();
            const unsigned int n_q_points = q_points.size();
            rhs_values.resize(fe.n_components() * n_q_points);


            // Interpolate at quadrature points:
#if 1
            // Interpolate values from the patch. This LEInteractor call could
            // be improved - we don't need to look up certain things on each
            // element
            static_assert(sizeof(Point<spacedim>) == sizeof(double) * spacedim,
                          "FORTRAN routines assume we are packed");
            const auto position_data =
              reinterpret_cast<const double *>(q_points.data());

            std::fill(rhs_values.begin(), rhs_values.end(), 0.0);
            IBTK::LEInteractor::interpolate(rhs_values.data(),
                                            rhs_values.size(),
                                            fe.n_components(),
                                            position_data,
                                            q_points.size() * spacedim,
                                            spacedim,
                                            patch_data,
                                            patch,
                                            patch->getBox(),
                                            kernel_name);
#else
            std::fill(rhs_values.begin(), rhs_values.end(), 1.0);
#endif

            for (unsigned int qp_n = 0; qp_n < n_q_points; ++qp_n)
              {
                if (fe.n_components() == 1)
                  {
                    AssertIndexRange(qp_n, rhs_values.size());
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        cell_rhs[i] += rhs_fe_values.shape_value(i, qp_n) *
                                       rhs_values[qp_n] *
                                       rhs_fe_values.JxW(qp_n);
                      }
                  }
                else if (fe.n_components() == spacedim)
                  {
                    Tensor<1, spacedim> qp;
                    for (unsigned int d = 0; d < spacedim; ++d)
                      {
                        AssertIndexRange(qp_n * spacedim + d,
                                         rhs_values.size());
                        qp[d] = rhs_values[qp_n * spacedim + d];
                      }

                    // TODO - this only works with primitive elements
                    // TODO - perhaps its worth unrolling this loop?
                    // dofs_per_cell is probably 12, 30, or 45.
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const unsigned int component =
                          fe.system_to_component_index(i).first;
                        cell_rhs[i] += rhs_fe_values.shape_value(i, qp_n) *
                                       qp[component] * rhs_fe_values.JxW(qp_n);
                      }
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }
              }

            rhs.add(dof_indices, cell_rhs);
          }
      }
  }



  template <int dim, int spacedim>
  void
  compute_projection_rhs(const std::string &                 kernel_name,
                         const int                           data_idx,
                         const PatchMap<dim, spacedim> &     patch_map,
                         const Mapping<dim, spacedim> &      position_mapping,
                         const std::vector<unsigned char> &  quadrature_indices,
                         const std::vector<Quadrature<dim>> &quadratures,
                         const DoFHandler<dim, spacedim> &   dof_handler,
                         const Mapping<dim, spacedim> &      mapping,
                         Vector<double> &                    rhs)
  {
#define ARGUMENTS                                                         \
  kernel_name, data_idx, patch_map, position_mapping, quadrature_indices, \
    quadratures, dof_handler, mapping, rhs
    if (patch_map.size() != 0)
      {
        auto patch_data = patch_map.get_patch(0)->getPatchData(data_idx);
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
                         std::vector<value_type> &      values)
  {
    Assert(false, ExcNotImplemented());
  }

  template <int dim, int spacedim>
  void
  compute_values_generic(const FEValues<dim, spacedim> &fe_values,
                         const Vector<double>           fe_solution,
                         std::vector<double> &          values)
  {
    fe_values.get_function_values(fe_solution, values);
  }

  template <int dim, int spacedim>
  void
  compute_values_generic(const FEValues<dim, spacedim> &   fe_values,
                         const Vector<double>              fe_solution,
                         std::vector<Tensor<1, spacedim>> &values)
  {
    const FEValuesExtractors::Vector vec(0);
    fe_values[vec].get_function_values(fe_solution, values);
  }

  template <int dim, int spacedim, typename value_type, typename patch_type>
  void
  compute_spread_internal(const std::string &               kernel_name,
                          const int                         data_idx,
                          PatchMap<dim, spacedim> &         patch_map,
                          const Mapping<dim, spacedim> &    position_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<dim>> &quadratures,
                          const DoFHandler<dim, spacedim> &   dof_handler,
                          const Mapping<dim, spacedim> &      mapping,
                          const Vector<double> &              solution)
  {
    check_quadratures(quadrature_indices,
                      quadratures,
                      dof_handler.get_triangulation());
    const FiniteElement<dim, spacedim> &fe = dof_handler.get_fe();

    // We probably don't need more than 16 quadrature rules
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_position_fe_values;
    boost::container::small_vector<std::unique_ptr<FEValues<dim, spacedim>>, 16>
      all_solution_fe_values;
    for (const Quadrature<dim> &quad : quadratures)
      {
        all_position_fe_values.emplace_back(
          std::make_unique<FEValues<dim, spacedim>>(
            position_mapping, fe, quad, update_quadrature_points));
        all_solution_fe_values.emplace_back(
          std::make_unique<FEValues<dim, spacedim>>(
            mapping, fe, quad, update_JxW_values | update_values));
      }

    std::vector<value_type> solution_values;

    for (unsigned int patch_n = 0; patch_n < patch_map.size(); ++patch_n)
      {
        auto                      patch      = patch_map.get_patch(patch_n);
        tbox::Pointer<patch_type> patch_data = patch->getPatchData(data_idx);
        Assert(patch_data, ExcMessage("Type mismatch"));
        check_depth(patch_data, fe);

        auto       iter = patch_map.begin(patch_n, dof_handler);
        const auto end  = patch_map.end(patch_n, dof_handler);
        for (; iter != end; ++iter)
          {
            const auto cell = *iter;
            const auto quad_index =
              quadrature_indices[cell->active_cell_index()];

            // Reinitialize:
            FEValues<dim, spacedim> &solution_fe_values =
              *all_solution_fe_values[quad_index];
            FEValues<dim, spacedim> &position_fe_values =
              *all_position_fe_values[quad_index];
            solution_fe_values.reinit(cell);
            position_fe_values.reinit(cell);
            Assert(solution_fe_values.get_quadrature() ==
                     position_fe_values.get_quadrature(),
                   ExcFDLInternalError());

            const std::vector<Point<spacedim>> &q_points =
              position_fe_values.get_quadrature_points();
            const unsigned int n_q_points = q_points.size();
            solution_values.resize(n_q_points);

            // get forces:
            std::fill(solution_values.begin(),
                      solution_values.end(),
                      value_type());
            compute_values_generic(solution_fe_values,
                                   solution,
                                   solution_values);
            for (unsigned int qp = 0; qp < n_q_points; ++qp)
              solution_values[qp] *= solution_fe_values.JxW(qp);

            // TODO reimplement zeroExteriorValues here

            // spread at quadrature points:
            static_assert(sizeof(Point<spacedim>) == sizeof(double) * spacedim,
                          "FORTRAN routines assume we are packed");
            const auto position_data =
              reinterpret_cast<const double *>(q_points.data());
            // the number of components is determined at run time so use a
            // normal assertion
            AssertThrow(sizeof(value_type) ==
                          sizeof(double) * fe.n_components(),
                        ExcMessage("FORTRAN routines assume we are packed"));
            const auto solution_data =
              reinterpret_cast<const double *>(solution_values.data());

            IBTK::LEInteractor::spread(patch_data,
                                       solution_data,
                                       solution_values.size() *
                                         fe.n_components(),
                                       fe.n_components(),
                                       position_data,
                                       n_q_points * spacedim,
                                       spacedim,
                                       patch,
                                       patch->getBox(),
                                       kernel_name);
          }
      }
  }



  template <int dim, int spacedim>
  void
  compute_spread(const std::string &                 kernel_name,
                 const int                           data_idx,
                 PatchMap<dim, spacedim> &           patch_map,
                 const Mapping<dim, spacedim> &      position_mapping,
                 const std::vector<unsigned char> &  quadrature_indices,
                 const std::vector<Quadrature<dim>> &quadratures,
                 const DoFHandler<dim, spacedim> &   dof_handler,
                 const Mapping<dim, spacedim> &      mapping,
                 const Vector<double> &              solution)
  {
#define ARGUMENTS                                                         \
  kernel_name, data_idx, patch_map, position_mapping, quadrature_indices, \
    quadratures, dof_handler, mapping, solution
    if (patch_map.size() != 0)
      {
        auto patch_data = patch_map.get_patch(0)->getPatchData(data_idx);
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
                          const Mapping<NDIM - 1, NDIM> &   position_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<NDIM - 1>> &quadratures);

  template void
  count_quadrature_points(const int                         qp_data_idx,
                          PatchMap<NDIM, NDIM> &            patch_map,
                          const Mapping<NDIM, NDIM> &       position_mapping,
                          const std::vector<unsigned char> &quadrature_indices,
                          const std::vector<Quadrature<NDIM>> &quadratures);

  template void
  compute_projection_rhs(const std::string &               kernel_name,
                         const int                         data_idx,
                         const PatchMap<NDIM - 1, NDIM> &  patch_map,
                         const Mapping<NDIM - 1, NDIM> &   position_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM - 1>> &quadratures,
                         const DoFHandler<NDIM - 1, NDIM> &       dof_handler,
                         const Mapping<NDIM - 1, NDIM> &          mapping,
                         Vector<double> &                         rhs);

  template void
  compute_projection_rhs(const std::string &               kernel_name,
                         const int                         data_idx,
                         const PatchMap<NDIM> &            patch_map,
                         const Mapping<NDIM> &             position_mapping,
                         const std::vector<unsigned char> &quadrature_indices,
                         const std::vector<Quadrature<NDIM>> &quadratures,
                         const DoFHandler<NDIM> &             dof_handler,
                         const Mapping<NDIM> &                mapping,
                         Vector<double> &                     rhs);

  template void
  compute_spread(const std::string &                      kernel_name,
                 const int                                data_idx,
                 PatchMap<NDIM - 1, NDIM> &               patch_map,
                 const Mapping<NDIM - 1, NDIM> &          position_mapping,
                 const std::vector<unsigned char> &       quadrature_indices,
                 const std::vector<Quadrature<NDIM - 1>> &quadratures,
                 const DoFHandler<NDIM - 1, NDIM> &       dof_handler,
                 const Mapping<NDIM - 1, NDIM> &          mapping,
                 const Vector<double> &                   solution);

  template void
  compute_spread(const std::string &                  kernel_name,
                 const int                            data_idx,
                 PatchMap<NDIM, NDIM> &               patch_map,
                 const Mapping<NDIM, NDIM> &          position_mapping,
                 const std::vector<unsigned char> &   quadrature_indices,
                 const std::vector<Quadrature<NDIM>> &quadratures,
                 const DoFHandler<NDIM, NDIM> &       dof_handler,
                 const Mapping<NDIM, NDIM> &          mapping,
                 const Vector<double> &               solution);
} // namespace fdl
