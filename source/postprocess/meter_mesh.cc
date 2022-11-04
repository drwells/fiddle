#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>

#include <CartesianPatchGeometry.h>
#include <fiddle/postprocess/meter_mesh.h>

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
      setup_meter_tria(const std::vector<Point<2>> &          hull,
                       parallel::shared::Triangulation<1, 2> &tria,
                       const double /*target_element_area*/)
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

        tria.create_triangulation(hull, cell_data, SubCellData());
      }
#else
      void
      setup_meter_tria(const std::vector<Point<3>> &          convex_hull,
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
    const Mapping<dim, spacedim> &                    mapping,
    const DoFHandler<dim, spacedim> &                 position_dof_handler,
    const std::vector<Point<spacedim>> &              convex_hull,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
    const int                                         level_number,
    const LinearAlgebra::distributed::Vector<double> &position)
    : mapping(&mapping)
    , patch_hierarchy(patch_hierarchy)
    , level_number(level_number)
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
    reinit(position);
  }

  template <int dim, int spacedim>
  void
  MeterMesh<dim, spacedim>::reinit(
    const LinearAlgebra::distributed::Vector<double> &position)
  {
    // Reset the meter mesh according to the new position values:
    const std::vector<Tensor<1, spacedim>> position_values =
      point_values->evaluate(position);
    const std::vector<Point<spacedim>> positions(position_values.begin(),
                                                 position_values.end());

    double dx_0 = std::numeric_limits<double>::max();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      patch_hierarchy->getPatchLevel(level_number);
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
    internal::setup_meter_tria(positions, meter_tria, target_element_area);

    scalar_dof_handler.reinit(meter_tria);
    scalar_dof_handler.distribute_dofs(*scalar_fe);
    vector_dof_handler.reinit(meter_tria);
    vector_dof_handler.distribute_dofs(*vector_fe);

    ;
    const auto local_bboxes = compute_cell_bboxes<dim - 1, spacedim, float>(
      vector_dof_handler,
      meter_tria.get_reference_cells()[0]
        .template get_default_linear_mapping<dim - 1, spacedim>());
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
    LinearAlgebra::distributed::Vector<double> identity_position(
      vector_partitioner);
    VectorTools::interpolate(vector_dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             identity_position);

    // TODO we'll need some extra code to guarantee that the box containing
    // the meter mesh is on the finest level: alternatively, perhaps we can
    // 'fix' NodalInteraction so that it supports structures on multiple
    // levels. That shouldn't be too bad since we just need to figure out
    // which nodes are on the interiors of which patches (much easier than
    // splitting elements)
    nodal_interaction = std::make_unique<NodalInteraction<dim - 1, spacedim>>(
      meter_tria,
      all_bboxes,
      // this vector isn't read so we can skip it
      std::vector<float>(),
      patch_hierarchy,
      level_number,
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
    Assert(false, ExcFDLNotImplemented());
    return {};
  }


  template class MeterMesh<NDIM, NDIM>;

} // namespace fdl
