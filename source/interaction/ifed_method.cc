#include <fiddle/base/samrai_utilities.h>

#include <fiddle/grid/box_utilities.h>

#include <fiddle/interaction/elemental_interaction.h>
#include <fiddle/interaction/ifed_method.h>
#include <fiddle/interaction/interaction_utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/fe/mapping_fe_field.h>

#include <CellVariable.h>
#include <IntVector.h>

namespace fdl
{
  using namespace dealii;
  using namespace SAMRAI;

  template <int dim, int spacedim>
  IFEDMethod<dim, spacedim>::IFEDMethod(
    tbox::Pointer<tbox::Database>      input_db,
    std::vector<Part<dim, spacedim>> &&input_parts)
    : input_db(input_db)
    , parts(std::move(input_parts))
    , secondary_hierarchy("ifed::secondary_hierarchy",
                          input_db->getDatabase("GriddingAlgorithm"),
                          input_db->getDatabase("LoadBalancer"))
  {
    // TODO: set up parameters correctly instead of hardcoding
    const unsigned int n_points_1D = 2;
    const double density = 1.0;
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      interactions.emplace_back(new ElementalInteraction<dim, spacedim>(n_points_1D, density));
  }

  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::initializePatchHierarchy(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> hierarchy,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/,
    int /*u_data_index*/,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      & /*u_synch_scheds*/,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      & /*u_ghost_fill_scheds*/,
    int /*integrator_step*/,
    double /*init_data_time*/,
    bool /*initial_time*/)
  {
    primary_hierarchy = hierarchy;

    secondary_hierarchy.reinit(primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy);

    reinit_interactions();
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::interpolateVelocity(
    int u_data_index,
    const std::vector<tbox::Pointer<xfer::CoarsenSchedule<spacedim>>>
      &u_synch_scheds,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      &    u_ghost_fill_scheds,
    double data_time)
  {
    (void)u_data_index;
    (void)u_synch_scheds;
    (void)u_ghost_fill_scheds;
    (void)data_time;
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::spreadForce(
    int                               f_data_index,
    IBTK::RobinPhysBdryPatchStrategy *f_phys_bdry_op,
    const std::vector<tbox::Pointer<xfer::RefineSchedule<spacedim>>>
      &    f_prolongation_scheds,
    double data_time)
  {
    (void)f_data_index;
    (void)f_phys_bdry_op;
    (void)f_prolongation_scheds;
    (void)data_time;
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::applyGradientDetector(
    tbox::Pointer<hier::BasePatchHierarchy<spacedim>> hierarchy,
    int                                               level_number,
    double /*error_data_time*/,
    int tag_index,
    bool /*initial_time*/,
    bool /*uses_richardson_extrapolation_too*/)
  {
    // TODO: we should find a way to save the bboxes so they do not need to be
    // computed for each level that needs tagging - conceivably this could
    // happen in beginDataRedistribution() and the array can be cleared in
    // endDataRedistribution()
    for (const Part<dim, spacedim> &part : parts)
      {
        const DoFHandler<dim, spacedim> &dof_handler = part.get_dof_handler();
        MappingFEField<dim,
                       spacedim,
                       LinearAlgebra::distributed::Vector<double>>
                   mapping(dof_handler, part.get_position());
        const auto local_bboxes =
          compute_cell_bboxes<dim, spacedim, float>(dof_handler, mapping);
        // Like most other things this only works with p::S::T now
        const auto &tria =
          dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> &>(
            part.get_triangulation());
        const auto global_bboxes =
          collect_all_active_cell_bboxes(tria, local_bboxes);
        tbox::Pointer<hier::PatchLevel<spacedim>> patch_level =
          hierarchy->getPatchLevel(level_number);
        Assert(patch_level, ExcNotImplemented());
        tag_cells(global_bboxes, tag_index, patch_level);
      }
  }

  template <int dim, int spacedim>
  void IFEDMethod<dim, spacedim>::beginDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    // TODO - calculate a nonzero workload using the secondary hierarchy
    const int ln = primary_hierarchy->getFinestLevelNumber();
    tbox::Pointer<hier::PatchLevel<spacedim>> level =
      primary_hierarchy->getPatchLevel(ln);
    if (!level->checkAllocated(lagrangian_workload_current_index))
      level->allocatePatchData(lagrangian_workload_current_index);

    auto ops =
      extract_hierarchy_data_ops(lagrangian_workload_var, primary_hierarchy);
    ops->resetLevels(ln, ln);
    ops->setToScalar(lagrangian_workload_current_index, 0.0);
  }

  template <int dim, int spacedim>
  void IFEDMethod<dim, spacedim>::endDataRedistribution(
    tbox::Pointer<hier::PatchHierarchy<spacedim>> /*hierarchy*/,
    tbox::Pointer<mesh::GriddingAlgorithm<spacedim>> /*gridding_alg*/)
  {
    secondary_hierarchy.reinit(primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy->getFinestLevelNumber(),
                               primary_hierarchy,
                               lagrangian_workload_current_index);

    reinit_interactions();
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::registerEulerianVariables()
  {
    // we need ghosts for CONSERVATIVE_LINEAR_REFINE
    const hier::IntVector<spacedim> ghosts = 1;
    lagrangian_workload_var =
      new pdat::CellVariable<spacedim, double>("::lagrangian_workload");
    registerVariable(lagrangian_workload_current_index,
                     lagrangian_workload_new_index,
                     lagrangian_workload_scratch_index,
                     lagrangian_workload_var,
                     ghosts,
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE");
  }


  template <int dim, int spacedim>
  const hier::IntVector<spacedim> &
  IFEDMethod<dim, spacedim>::getMinimumGhostCellWidth() const
  {
    // Like elsewhere, we are hard-coding in bspline 3 for now
    const std::string kernel_name = "BSPLINE_3";
    const int         ghost_width =
      IBTK::LEInteractor::getMinimumGhostWidth(kernel_name);
    static hier::IntVector<spacedim> gcw;
    for (int i = 0; i < spacedim; ++i)
      gcw[i] = ghost_width;
    return gcw;
  }



  template <int dim, int spacedim>
  void
  IFEDMethod<dim, spacedim>::reinit_interactions()
  {
    for (unsigned int part_n = 0; part_n < parts.size(); ++part_n)
      {
        const Part<dim, spacedim> &part = parts[part_n];

        const auto &tria =
          dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> &>(
            part.get_triangulation());
        const DoFHandler<dim, spacedim> &dof_handler = part.get_dof_handler();
        MappingFEField<dim,
                       spacedim,
                       LinearAlgebra::distributed::Vector<double>>
                   mapping(dof_handler, part.get_position());
        const auto local_bboxes =
          compute_cell_bboxes<dim, spacedim, float>(dof_handler, mapping);

        const auto global_bboxes =
          collect_all_active_cell_bboxes(tria, local_bboxes);

        interactions[part_n]->reinit(tria,
                                     global_bboxes,
                                     secondary_hierarchy.d_secondary_hierarchy,
                                     primary_hierarchy->getFinestLevelNumber());
      }
  }


  template class IFEDMethod<NDIM - 1, NDIM>;
  template class IFEDMethod<NDIM, NDIM>;
} // namespace fdl
