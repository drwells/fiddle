#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/postprocess/meter_mesh.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <CartesianPatchGeometry.h>
#include <tbox/InputManager.h>

#include <cmath>
#include <limits>

namespace fdl
{
  namespace internal
  {
    namespace
    {
      // avoid "defined but not used" warnings by using NDIM
#if NDIM == 2
      void
      setup_meter_tria(const std::vector<Point<2>>           &hull,
                       parallel::shared::Triangulation<1, 2> &tria,
                       const double target_element_area)
      {
        // TODO: make sure that we add enough extra elements to each line
        // segment to meet the target_element_area requirement.
        Assert(hull.size() > 1, ExcFDLInternalError());
        std::vector<CellData<1>> cell_data;

        for (unsigned int vertex_n = 0; vertex_n < hull.size() - 1; ++vertex_n)
          {
            cell_data.emplace_back();
            cell_data.back().vertices[0] = vertex_n;
            cell_data.back().vertices[1] = vertex_n + 1;
          }

        std::vector<unsigned int> all_vertices;
        std::vector<Point<2>>     vertices = hull;
        SubCellData               sub_cell_data;
        GridTools::delete_duplicated_vertices(vertices,
                                              cell_data,
                                              sub_cell_data,
                                              all_vertices);
        GridTools::consistently_order_cells(cell_data);
        tria.create_triangulation(vertices, cell_data, sub_cell_data);
      }
#else
      void
      setup_meter_tria(const std::vector<Point<3>>           &convex_hull,
                       parallel::shared::Triangulation<2, 3> &tria,
                       const double target_element_area)
      {
        Assert(convex_hull.size() > 2, ExcFDLInternalError());

        Triangle::AdditionalData additional_data;
        additional_data.target_element_area = target_element_area;
        // TODO - use the normal vector returned by setup_meter_tria() somewhere
        setup_planar_meter_mesh(convex_hull, tria, additional_data);
      }
#endif
    } // namespace
  }   // namespace internal

  template <int dim, int spacedim>
  MeterMesh<dim, spacedim>::MeterMesh(
    const Mapping<dim, spacedim>                     &mapping,
    const DoFHandler<dim, spacedim>                  &position_dof_handler,
    const std::vector<Point<spacedim>>               &convex_hull,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
    : mapping(&mapping)
    , patch_hierarchy(patch_hierarchy)
    , point_values(std::make_unique<PointValues<spacedim, dim, spacedim>>(
        mapping,
        position_dof_handler,
        convex_hull))
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim - 1, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim - 1, spacedim>>(1))
    , vector_fe(
        std::make_unique<FESystem<dim - 1, spacedim>>(*scalar_fe, spacedim))
  {
    // TODO: assert congruity between position_dof_handler.get_communicator()
    // and SAMRAI_MPI::getCommunicator()
    reinit(position, velocity);
  }

  template <int dim, int spacedim>
  MeterMesh<dim, spacedim>::MeterMesh(
    const std::vector<Point<spacedim>>               &convex_hull,
    const std::vector<Tensor<1, spacedim>>           &velocity,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim - 1, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim - 1, spacedim>>(1))
    , vector_fe(
        std::make_unique<FESystem<dim - 1, spacedim>>(*scalar_fe, spacedim))
  {
    reinit_tria(convex_hull);

    // TODO do something with the velocity
  }

  template <int dim, int spacedim>
  void
  MeterMesh<dim, spacedim>::reinit(
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
  {
    // Reset the meter mesh according to the new position values:
    const std::vector<Tensor<1, spacedim>> position_values =
      point_values->evaluate(position);
    const std::vector<Point<spacedim>> positions(position_values.begin(),
                                                 position_values.end());

    reinit_tria(positions);

    // TODO do something with the velocity
  }

  template <int dim, int spacedim>
  void
  MeterMesh<dim, spacedim>::reinit(
    const std::vector<Point<spacedim>> &    convex_hull,
    const std::vector<Tensor<1, spacedim>> &velocity)
  {
    reinit_tria(convex_hull);

    // TODO do something with the velocity
  }

  template <int dim, int spacedim>
  void
  MeterMesh<dim, spacedim>::reinit_tria(
    const std::vector<Point<spacedim>> &convex_hull)
  {
    double dx_0 = std::numeric_limits<double>::max();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      patch_hierarchy->getPatchLevel(patch_hierarchy->getFinestLevelNumber());
    Assert(level, ExcFDLInternalError());
    for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
      {
        tbox::Pointer<hier::Patch<spacedim>> patch = level->getPatch(p());
        const tbox::Pointer<geom::CartesianPatchGeometry<spacedim>> pgeom =
          patch->getPatchGeometry();
        dx_0 = std::min(dx_0,
                        *std::min_element(pgeom->getDx(),
                                          pgeom->getDx() + spacedim));
      }
    dx_0 = Utilities::MPI::min(dx_0, tbox::SAMRAI_MPI::getCommunicator());
    Assert(dx_0 != std::numeric_limits<double>::max(), ExcFDLInternalError());
    const double target_element_area = std::pow(dx_0, dim - 1);

    meter_tria.clear();
    internal::setup_meter_tria(convex_hull, meter_tria, target_element_area);

    meter_mapping = meter_tria.get_reference_cells()[0]
                      .template get_default_mapping<dim - 1, spacedim>(
                        scalar_fe->tensor_degree());
    meter_quadrature =
      QWitherdenVincentSimplex<dim - 1>(scalar_fe->tensor_degree() + 1);

    scalar_dof_handler.reinit(meter_tria);
    scalar_dof_handler.distribute_dofs(*scalar_fe);
    vector_dof_handler.reinit(meter_tria);
    vector_dof_handler.distribute_dofs(*vector_fe);

    const auto local_bboxes =
      compute_cell_bboxes<dim - 1, spacedim, float>(vector_dof_handler,
                                                    *meter_mapping);
    const auto all_bboxes =
      collect_all_active_cell_bboxes(meter_tria, local_bboxes);

    // Set up partitioners:
    {
      IndexSet vector_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(vector_dof_handler,
                                              vector_locally_relevant_dofs);
      vector_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        vector_dof_handler.locally_owned_dofs(),
        vector_locally_relevant_dofs,
        vector_dof_handler.get_communicator());

      IndexSet scalar_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(scalar_dof_handler,
                                              scalar_locally_relevant_dofs);
      scalar_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        scalar_dof_handler.locally_owned_dofs(),
        scalar_locally_relevant_dofs,
        scalar_dof_handler.get_communicator());
    }
    identity_position.reinit(vector_partitioner);
    VectorTools::interpolate(vector_dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             identity_position);
    identity_position.update_ghost_values();

    // 1e-6 is an arbitrary nonzero number which ensures that points on the
    // boundaries between patches end up in both (for the purposes of
    // computing interpolations) but minimizes the amount of work resulting
    // from double-counting. I suspect that any number larger than 1e-10 would
    // suffice.
    tbox::Pointer<tbox::Database> db = new tbox::InputDatabase("meter_mesh_db");
    db->putDouble("ghost_cell_fraction", 1e-6);
    nodal_interaction = std::make_unique<NodalInteraction<dim - 1, spacedim>>(
      db,
      meter_tria,
      all_bboxes,
      patch_hierarchy,
      std::make_pair(0, patch_hierarchy->getFinestLevelNumber()),
      vector_dof_handler,
      identity_position);
    nodal_interaction->add_dof_handler(scalar_dof_handler);
  }

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  MeterMesh<dim, spacedim>::mean_meter_velocity(const int          data_idx,
                                                const std::string &kernel_name)
  {
    Assert(false, ExcFDLNotImplemented());
    return {};
  }

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  MeterMesh<dim, spacedim>::mean_flux(const int          data_idx,
                                      const std::string &kernel_name)
  {
    Assert(false, ExcFDLNotImplemented());
    return {};
  }

  template <int dim, int spacedim>
  double
  MeterMesh<dim, spacedim>::mean_value(const int          data_idx,
                                       const std::string &kernel_name)
  {
    LinearAlgebra::distributed::Vector<double> interpolated_data(
      scalar_partitioner);
    auto transaction =
      nodal_interaction->compute_projection_rhs_start(kernel_name,
                                                      data_idx,
                                                      vector_dof_handler,
                                                      identity_position,
                                                      scalar_dof_handler,
                                                      *meter_mapping,
                                                      interpolated_data);
    transaction = nodal_interaction->compute_projection_rhs_intermediate(
      std::move(transaction));
    nodal_interaction->compute_projection_rhs_finish(std::move(transaction));
    interpolated_data.update_ghost_values();

    return VectorTools::compute_mean_value(*meter_mapping,
                                           scalar_dof_handler,
                                           meter_quadrature,
                                           interpolated_data,
                                           0);
  }


  template class MeterMesh<NDIM, NDIM>;

} // namespace fdl
