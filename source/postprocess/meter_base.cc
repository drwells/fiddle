#include <fiddle/base/exceptions.h>

#include <fiddle/grid/box_utilities.h>
#include <fiddle/grid/surface_tria.h>

#include <fiddle/postprocess/meter_base.h>

#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <ibtk/IndexUtilities.h>

#include <CartesianPatchGeometry.h>
#include <tbox/InputManager.h>

#include <cmath>
#include <limits>

namespace fdl
{
  template <int dim, int spacedim>
  MeterBase<dim, spacedim>::MeterBase(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
    , scalar_fe(std::make_unique<FE_SimplexP<dim, spacedim>>(1))
    , vector_fe(std::make_unique<FESystem<dim, spacedim>>(*scalar_fe, spacedim))
  {}

  template <int dim, int spacedim>
  MeterBase<dim, spacedim>::MeterBase(
    const Triangulation<dim, spacedim>           &tria,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
    : patch_hierarchy(patch_hierarchy)
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
  {
    AssertThrow(!tria.has_hanging_nodes(), ExcFDLNotImplemented());
    GridGenerator::flatten_triangulation(tria, meter_tria);

    if (tria.all_reference_cells_are_simplex())
      scalar_fe = std::make_unique<FE_SimplexP<dim, spacedim>>(1);
    else if (tria.all_reference_cells_are_hyper_cube())
      scalar_fe = std::make_unique<FE_Q<dim, spacedim>>(1);
    else
      AssertThrow(false,
                  ExcMessage("mixed meshes are not yet supported here."));

    vector_fe = std::make_unique<FESystem<dim, spacedim>>(*scalar_fe, spacedim);
  }

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::reinit_interaction()
  {
    // As the meter mesh is in absolute coordinates we can use a normal
    // mapping here
    const auto local_bboxes =
      compute_cell_bboxes<dim, spacedim, float>(get_vector_dof_handler(),
                                                get_mapping());
    const auto all_bboxes =
      collect_all_active_cell_bboxes(meter_tria, local_bboxes);

    // 1e-6 is an arbitrary nonzero number which ensures that points on the
    // boundaries between patches end up in both (for the purposes of
    // computing interpolations) but minimizes the amount of work resulting
    // from double-counting. I suspect that any number larger than 1e-10 would
    // suffice.
    tbox::Pointer<tbox::Database> db = new tbox::InputDatabase("meter_mesh_db");
    db->putDouble("ghost_cell_fraction", 1e-6);
    nodal_interaction = std::make_unique<NodalInteraction<dim, spacedim>>(
      db,
      meter_tria,
      all_bboxes,
      patch_hierarchy,
      std::make_pair(0, patch_hierarchy->getFinestLevelNumber()),
      get_vector_dof_handler(),
      identity_position);
    nodal_interaction->add_dof_handler(get_scalar_dof_handler());
  }

  template <int dim, int spacedim>
  bool
  MeterBase<dim, spacedim>::compute_vertices_inside_domain() const
  {
    tbox::Pointer<geom::CartesianGridGeometry<spacedim>> geom =
      patch_hierarchy->getGridGeometry();
    Assert(geom, ExcFDLInternalError());
    tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
      patch_hierarchy->getPatchLevel(0);
    Assert(patch_level, ExcFDLInternalError());

    bool vertices_inside_domain = true;
    for (const auto &vertex : get_triangulation().get_vertices())
      {
        const auto index =
          IBTK::IndexUtilities::getCellIndex(vertex,
                                             geom,
                                             hier::IntVector<spacedim>(1));
        vertices_inside_domain =
          vertices_inside_domain &&
          patch_level->getPhysicalDomain().contains(index);
      }

    return vertices_inside_domain;
  }

  template <int dim, int spacedim>
  void
  MeterBase<dim, spacedim>::reinit_dofs()
  {
    meter_mapping = meter_tria.get_reference_cells()[0]
                      .template get_default_mapping<dim, spacedim>(
                        scalar_fe->tensor_degree());
    if (meter_tria.all_reference_cells_are_simplex())
      meter_quadrature =
        QWitherdenVincentSimplex<dim>(scalar_fe->tensor_degree() + 1);
    else
      meter_quadrature = QGauss<dim>(scalar_fe->tensor_degree() + 1);

    scalar_dof_handler.reinit(meter_tria);
    scalar_dof_handler.distribute_dofs(*scalar_fe);
    vector_dof_handler.reinit(meter_tria);
    vector_dof_handler.distribute_dofs(*vector_fe);

    // Set up partitioners:
    const MPI_Comm comm = meter_tria.get_communicator();
    {
      IndexSet scalar_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(scalar_dof_handler,
                                              scalar_locally_relevant_dofs);
      scalar_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        scalar_dof_handler.locally_owned_dofs(),
        scalar_locally_relevant_dofs,
        comm);

      IndexSet vector_locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(vector_dof_handler,
                                              vector_locally_relevant_dofs);
      vector_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        vector_dof_handler.locally_owned_dofs(),
        vector_locally_relevant_dofs,
        comm);
    }
    identity_position.reinit(vector_partitioner);
    VectorTools::interpolate(vector_dof_handler,
                             Functions::IdentityFunction<spacedim>(),
                             identity_position);
    identity_position.update_ghost_values();
  }

  template class MeterBase<NDIM - 1, NDIM>;
  template class MeterBase<NDIM, NDIM>;
} // namespace fdl
