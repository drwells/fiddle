#include <fiddle/transfer/overlap_partitioning_tools.h>

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
  using namespace SAMRAI;

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

            // We just need an iterator to get a valid patch (so we can get a
            // valid type), so we are done at this point
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
        const unsigned int depth = f_data->getDepth();
        Assert(depth == f_fe.n_components(),
               ExcMessage("The depth of the SAMRAI variable should equal the "
                          "number of components of the finite element."));

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


  template <int dim, int spacedim>
  InteractionBase<dim, spacedim>::InteractionBase(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &     global_active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     p_hierarchy,
    const int                                             l_number,
    std::shared_ptr<IBTK::SAMRAIDataCache> e_data_cache)
    : native_tria(&n_tria)
    , patch_hierarchy(p_hierarchy)
    , level_number(l_number)
    , eulerian_data_cache(e_data_cache)
  {
    reinit(n_tria, global_active_cell_bboxes, p_hierarchy, l_number, e_data_cache);
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::reinit(
    const parallel::shared::Triangulation<dim, spacedim> &n_tria,
    const std::vector<BoundingBox<spacedim, float>> &     global_active_cell_bboxes,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>>     p_hierarchy,
    const int                                             l_number,
    std::shared_ptr<IBTK::SAMRAIDataCache> e_data_cache)
  {
    native_tria = &n_tria;
    patch_hierarchy = p_hierarchy;
    level_number = l_number;
    eulerian_data_cache = e_data_cache;

    // Check inputs
    Assert(global_active_cell_bboxes.size() == native_tria->n_active_cells(),
           ExcMessage("There should be a bounding box for each active cell"));
    Assert(patch_hierarchy,
           ExcMessage("The provided pointer to a patch hierarchy should not be "
                      "null."));
    AssertIndexRange(l_number, patch_hierarchy->getNumberOfLevels());
    Assert(eulerian_data_cache,
           ExcMessage("The provided shared pointer to an Eulerian data cache "
                      "should not be null."));

    const auto patches = extract_patches(
      patch_hierarchy->getPatchLevel(level_number));
    // TODO we need to make extra ghost cell fraction a parameter
    const std::vector<BoundingBox<spacedim>> patch_bboxes =
    compute_patch_bboxes(patches, 1.0);
    BoxIntersectionPredicate<dim, spacedim> predicate(
      global_active_cell_bboxes, patch_bboxes, *native_tria);
    overlap_tria.reinit(*native_tria, predicate);

    std::vector<BoundingBox<spacedim, float>> overlap_bboxes;
    for (const auto &cell : overlap_tria.active_cell_iterators())
      {
        auto native_cell = overlap_tria.get_native_cell(cell);
        overlap_bboxes.push_back(
          global_active_cell_bboxes[native_cell->active_cell_index()]);
      }

    // TODO add the ghost cell width as an input argument to this class
    patch_map.reinit(patches, 1.0, overlap_tria, overlap_bboxes);

    // Some other things that should be established at this point:
    //
    // cell_index_scatter: we can figure out the indices by looping over the
    //     triangulations and recording active cell indices
    //
    // I think our nonuniform partitioner class can actually be used here
    // instead for cell_index_scatter.
  }



  template <int dim, int spacedim>
  DoFHandler<dim, spacedim> &
  InteractionBase<dim, spacedim>::get_overlap_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    auto iter = std::find(native_dof_handlers.begin(), native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return *overlap_dof_handlers[iter - native_dof_handlers.begin()];
  }



  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  InteractionBase<dim, spacedim>::get_overlap_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler) const
  {
    auto iter = std::find(native_dof_handlers.begin(), native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return *overlap_dof_handlers[iter - native_dof_handlers.begin()];
  }



  template <int dim, int spacedim>
  Scatter<double> &
  InteractionBase<dim, spacedim>::get_scatter(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    auto iter = std::find(native_dof_handlers.begin(), native_dof_handlers.end(),
                          &native_dof_handler);
    AssertThrow(iter != native_dof_handlers.end(),
                ExcMessage("The provided dof handler must already be "
                           "registered with this class."));
    return scatters[iter - native_dof_handlers.begin()];
  }



  template <int dim, int spacedim>
  void
  InteractionBase<dim, spacedim>::add_dof_handler(
    const DoFHandler<dim, spacedim> &native_dof_handler)
  {
    AssertThrow(&native_dof_handler.get_triangulation() == native_tria,
                ExcMessage("The DoFHandler must use the underlying native "
                           "triangulation."));
    const auto ptr = &native_dof_handler;
    if (std::find(native_dof_handlers.begin(), native_dof_handlers.end(), ptr)
        != native_dof_handlers.end())
    {
      native_dof_handlers.emplace_back(ptr);
      // TODO - implement a move ctor for DH in deal.II
      overlap_dof_handlers.emplace_back(
        std::make_unique<DoFHandler<dim, spacedim>>(overlap_tria));
      auto &overlap_dof_handler = *overlap_dof_handlers.back();
      overlap_dof_handler.distribute_dofs(native_dof_handler.get_fe_collection());

      const std::vector<types::global_dof_index> overlap_to_native_dofs =
        compute_overlap_to_native_dof_translation(overlap_tria,
                                                  overlap_dof_handler,
                                                  native_dof_handler);
      scatters.emplace_back(overlap_to_native_dofs,
                            native_dof_handler.locally_owned_dofs(),
                            native_tria->get_communicator());
    }
  }



  template <int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_start(
    const int                                         f_data_idx,
    const QuadratureFamily<dim> &                     quad_family,
    const std::vector<unsigned char> &                quad_indices,
    const DoFHandler<dim, spacedim> &                 X_dof_handler,
    const LinearAlgebra::distributed::Vector<double> &X,
    const DoFHandler<dim, spacedim> &                 F_dof_handler,
    const Mapping<dim, spacedim> &                    F_mapping,
    LinearAlgebra::distributed::Vector<double> &      F_rhs)
  {
    auto t_ptr = std::make_unique<Transaction<dim, spacedim>>();

    Transaction<dim, spacedim> &transaction = *t_ptr;
    // set up everything we will need later
    transaction.current_f_data_idx  = f_data_idx;
    transaction.quad_family         = &quad_family;
    transaction.native_quad_indices = quad_indices;
    transaction.overlap_quad_indices.resize(overlap_tria.n_active_cells());

    // Setup X info:
    transaction.native_X_dof_handler = &X_dof_handler;
    transaction.native_X             = &X;
    transaction.overlap_X_vec.reinit(
      get_overlap_dof_handler(X_dof_handler).n_dofs());

    // Setup F info:
    transaction.native_F_dof_handler = &F_dof_handler;
    transaction.F_mapping            = &F_mapping;
    transaction.native_F_rhs         = &F_rhs;
    transaction.overlap_F_rhs.reinit(
      get_overlap_dof_handler(F_dof_handler).n_dofs());

    // Setup state:
    transaction.next_state = Transaction<dim, spacedim>::State::Intermediate;
    transaction.operation  = Transaction<dim, spacedim>::Operation::Interpolation;

    // OK, now start scattering:
    Scatter<double> &X_scatter = get_scatter(X_dof_handler);

    // TODO we really need a good way to get channels at this point so people
    // can start multiple scatters at once
    X_scatter.global_to_overlap_start(*transaction.native_X,
                                      0,
                                      transaction.overlap_X_vec);

    // TODO scatter quad indices too

    return t_ptr;
  }



  template<int dim, int spacedim>
  std::unique_ptr<TransactionBase>
  InteractionBase<dim, spacedim>::compute_projection_rhs_intermediate(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation == Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state == Transaction<dim, spacedim>::State::Intermediate),
           ExcMessage("Transaction state should be Intermediate"));

    Scatter<double> &X_scatter = get_scatter(*trans.native_X_dof_handler);

    X_scatter.global_to_overlap_finish(*trans.native_X,
                                       trans.overlap_X_vec);

    trans.next_state = Transaction<dim, spacedim>::State::Finish;

    return t_ptr;
  }



  template <int dim, int spacedim>
  void InteractionBase<dim, spacedim>::compute_projection_rhs_finish(
    std::unique_ptr<TransactionBase> t_ptr)
  {
    auto &trans = dynamic_cast<Transaction<dim, spacedim> &>(*t_ptr);
    Assert((trans.operation == Transaction<dim, spacedim>::Operation::Interpolation),
           ExcMessage("Transaction operation should be Interpolation"));
    Assert((trans.next_state == Transaction<dim, spacedim>::State::Finish),
           ExcMessage("Transaction state should be Finish"));

    Scatter<double> &F_scatter = get_scatter(*trans.native_F_dof_handler);
    F_scatter.overlap_to_global_finish(trans.overlap_F_rhs,
                                       VectorOperation::add,
                                       *trans.native_F_rhs);
    trans.next_state = Transaction<dim, spacedim>::State::Done;
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

  template class InteractionBase<NDIM - 1, NDIM>;
  template class InteractionBase<NDIM, NDIM>;
} // namespace fdl
