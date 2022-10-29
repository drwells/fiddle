#include <fiddle/postprocess/meter_mesh.h>

namespace fdl
{
  namespace internal
  {
    namespace
    {
      void
      setup_meter_tria(const std::vector<Point<2>> &          hull,
                       parallel::shared::Triangulation<1, 2> &tria,
                       const double /*target_element_area*/)
      {
        // Somewhat simpler case: just connect the dots.
        Assert(hull.size() > 1, ExcFDLInternalError());
        std::vector<CellData<1>> cell_data;

        for (unsigned int vertex_n = 0; vertex_n < hull.size() - 1; ++vertex_n)
          {
            cell_data.emplace_back();
            cell_data.back().vertices[0] = vertex_n;
            cell_data.back().vertices[1] = vertex_n + 1;
          }

        tria.make_triangulation(hull, cell_data, SubCellData());
      }

      void
      setup_meter_tria(const std::vector<Point<3>> &          convex_hull,
                       parallel::shared::Triangulation<1, 2> &tria,
                       const double target_element_area)
      {
        Assert(hull.size() > 2, ExcFDLInternalError());

        Triangle::AdditionalData additional_data;
        additional_data.target_element_area = target_element_area;
        triangulate_convex(hull_vertices, tria, additional_data);
      }
    } // namespace
  }   // namespace internal

  template <int dim, int spacedim>
  MeterMesh::MeterMesh(
    const Mapping<dim, spacedim> &                    mapping,
    const DoFHandler<dim, spacedim> &                 position_dof_handler,
    const std::vector<Point<spacedim>> &              convex_hull,
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> patch_hierarchy,
    const int                                         level_number,
    const LinearAlgebra::distributed::Vector<double> &position,
    const LinearAlgebra::distributed::Vector<double> &velocity)
    : mapping(&mapping)
    , position_dof_handler(&position_dof_handler)
    , patch_hierarchy(patch_hierarchy)
    , level_number(level_number)
    , point_values(std::make_unique<PointValues<spacedim, dim, spacedim>>(
        mapping,
        position_dof_handler,
        convex_hull))
    , meter_tria(tbox::SAMRAI_MPI::getCommunicator(),
                 Triangulation<dim, spacedim>::MeshSmoothing::none,
                 true)
  {
    const std::vector<Tensor<1, spacedim>> position_values =
      point_values->evaluate(position);
    const std::vector<Point<spacedim>> positions(position_values.begin(),
                                                 position_values.end());


    // TODO: determine dx here
    
    setup_meter_tria(positions, meter_tria, target_element_area);
  }
} // namespace fdl
