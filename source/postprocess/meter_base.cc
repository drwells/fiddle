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
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <ibtk/IndexUtilities.h>

#include <CartesianPatchGeometry.h>
#include <PatchGeometry.h>

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
  {}

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

  template class MeterBase<NDIM - 1, NDIM>;
  template class MeterBase<NDIM, NDIM>;
} // namespace fdl
