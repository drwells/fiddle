#include <fiddle/base/exceptions.h>
#include <fiddle/base/samrai_utilities.h>

#include <CellData.h>
#include <CellVariable.h>
#include <EdgeData.h>
#include <EdgeVariable.h>
#include <HierarchyCellDataOpsReal.h>
#include <HierarchyDataOpsReal.h>
#include <HierarchyEdgeDataOpsReal.h>
#include <HierarchyNodeDataOpsReal.h>
#include <HierarchySideDataOpsReal.h>
#include <NodeData.h>
#include <NodeVariable.h>
#include <SideData.h>
#include <SideVariable.h>

namespace fdl
{
  using namespace SAMRAI;

  /**
   * Another helper function since MultiblockPatchLevel and PatchLevel have
   * slightly different interfaces
   */
  template <int spacedim>
  std::vector<tbox::Pointer<hier::Patch<spacedim>>>
  extract_patches(
    tbox::Pointer<hier::BasePatchLevel<spacedim>> base_patch_level)
  {
    tbox::Pointer<hier::PatchLevel<spacedim>> patch_level(base_patch_level);
    tbox::Pointer<hier::MultiblockPatchLevel<spacedim>> multi_patch_level(
      base_patch_level);

    std::vector<tbox::Pointer<hier::Patch<spacedim>>> result;
    auto get_patches = [&](tbox::Pointer<hier::PatchLevel<spacedim>> level) {
      for (typename hier::PatchLevel<spacedim>::Iterator p(level); p; p++)
        result.emplace_back(level->getPatch(p()));
    };

    if (patch_level)
      {
        get_patches(patch_level);
      }
    else if (multi_patch_level)
      {
        for (int block_n = 0; block_n < multi_patch_level->getNumberOfBlocks();
             ++block_n)
          {
            get_patches(multi_patch_level->getPatchLevelForBlock(block_n));
          }
      }
    else
      {
        Assert(false, ExcFDLNotImplemented());
      }

    return result;
  }

  /**
   * Helper function for extracting locally owned patches from a patch level.
   */
  template <int spacedim>
  std::vector<tbox::Pointer<hier::Patch<spacedim>>>
  extract_patches(tbox::Pointer<hier::PatchLevel<spacedim>> patch_level)
  {
    // Downcast and move to the more general function
    tbox::Pointer<hier::BasePatchLevel<spacedim>> base_patch_level(patch_level);
    return extract_patches(base_patch_level);
  }

  template <int spacedim>
  std::pair<SAMRAIPatchType, SAMRAIFieldType>
  extract_types(const tbox::Pointer<hier::PatchData<spacedim>> &p)
  {
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, int>>(p))
      return {SAMRAIPatchType::Edge, SAMRAIFieldType::Int};
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, float>>(p))
      return {SAMRAIPatchType::Edge, SAMRAIFieldType::Float};
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, double>>(p))
      return {SAMRAIPatchType::Edge, SAMRAIFieldType::Double};

    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, int>>(p))
      return {SAMRAIPatchType::Cell, SAMRAIFieldType::Int};
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, float>>(p))
      return {SAMRAIPatchType::Cell, SAMRAIFieldType::Float};
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, double>>(p))
      return {SAMRAIPatchType::Cell, SAMRAIFieldType::Double};

    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, int>>(p))
      return {SAMRAIPatchType::Node, SAMRAIFieldType::Int};
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, float>>(p))
      return {SAMRAIPatchType::Node, SAMRAIFieldType::Float};
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, double>>(p))
      return {SAMRAIPatchType::Node, SAMRAIFieldType::Double};

    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, int>>(p))
      return {SAMRAIPatchType::Side, SAMRAIFieldType::Int};
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, float>>(p))
      return {SAMRAIPatchType::Side, SAMRAIFieldType::Float};
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, double>>(p))
      return {SAMRAIPatchType::Side, SAMRAIFieldType::Double};

    AssertThrow(false, ExcFDLNotImplemented());
    return {};
  }

  template <int spacedim>
  int
  extract_depth(const tbox::Pointer<hier::PatchData<spacedim>> &p)
  {
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, int>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, float>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, double>>(p))
      return p2->getDepth();

    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, int>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, float>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, double>>(p))
      return p2->getDepth();

    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, int>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, float>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, double>>(p))
      return p2->getDepth();

    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, int>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, float>>(p))
      return p2->getDepth();
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, double>>(p))
      return p2->getDepth();

    AssertThrow(false, ExcFDLNotImplemented());
    return {};
  }

  template <int spacedim, typename field_type>
  void
  fill_all(tbox::Pointer<hier::PatchData<spacedim>> p, const field_type value)
  {
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, int>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, float>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::EdgeData<spacedim, double>>(p))
      {
        p2->fillAll(value);
        return;
      }

    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, int>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, float>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::CellData<spacedim, double>>(p))
      {
        p2->fillAll(value);
        return;
      }

    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, int>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, float>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::NodeData<spacedim, double>>(p))
      {
        p2->fillAll(value);
        return;
      }

    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, int>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, float>>(p))
      {
        p2->fillAll(value);
        return;
      }
    if (auto p2 = tbox::Pointer<pdat::SideData<spacedim, double>>(p))
      {
        p2->fillAll(value);
        return;
      }

    AssertThrow(false, ExcFDLNotImplemented());
  }

  template <int spacedim, typename field_type>
  void
  fill_all(tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy,
           const int                                     data_index,
           const int                                     coarsest_level_number,
           const int                                     finest_level_number,
           const field_type                              value,
           const bool                                    interior_only)
  {
    Assert(interior_only == false, ExcFDLNotImplemented());
    for (int ln = coarsest_level_number; ln <= finest_level_number; ++ln)
      {
        auto patches = extract_patches(patch_hierarchy->getPatchLevel(ln));

        for (auto &patch : patches)
          {
            fill_all(patch->getPatchData(data_index), value);
          }
      }
  }

  template <int spacedim>
  tbox::Pointer<math::HierarchyDataOpsReal<spacedim, double>>
  extract_hierarchy_data_ops(
    const tbox::Pointer<hier::Variable<spacedim>> p,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy)
  {
    if (auto p2 = tbox::Pointer<pdat::EdgeVariable<spacedim, double>>(p))
      return new math::HierarchyEdgeDataOpsReal<spacedim, double>(
        patch_hierarchy);
    else if (auto p2 = tbox::Pointer<pdat::CellVariable<spacedim, double>>(p))
      return new math::HierarchyCellDataOpsReal<spacedim, double>(
        patch_hierarchy);
    else if (auto p2 = tbox::Pointer<pdat::NodeVariable<spacedim, double>>(p))
      return new math::HierarchyNodeDataOpsReal<spacedim, double>(
        patch_hierarchy);
    else if (auto p2 = tbox::Pointer<pdat::SideVariable<spacedim, double>>(p))
      return new math::HierarchySideDataOpsReal<spacedim, double>(
        patch_hierarchy);

    AssertThrow(false, ExcFDLNotImplemented());
  }

  // instantiations:
  template std::vector<tbox::Pointer<hier::Patch<NDIM>>>
  extract_patches(tbox::Pointer<hier::BasePatchLevel<NDIM>> patch_level);

  template std::vector<tbox::Pointer<hier::Patch<NDIM>>>
  extract_patches(tbox::Pointer<hier::PatchLevel<NDIM>> patch_level);

  template std::pair<SAMRAIPatchType, SAMRAIFieldType>
  extract_types(const tbox::Pointer<hier::PatchData<NDIM>> &p);

  template int
  extract_depth(const tbox::Pointer<hier::PatchData<NDIM>> &p);

  template void
  fill_all(tbox::Pointer<hier::PatchData<NDIM>> p, const int value);

  template void
  fill_all(tbox::Pointer<hier::PatchData<NDIM>> p, const float value);

  template void
  fill_all(tbox::Pointer<hier::PatchData<NDIM>> p, const double value);

  template void
  fill_all(tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy,
           const int                                 data_index,
           const int                                 coarsest_level_number,
           const int                                 finest_level_number,
           const int                                 value,
           const bool                                interior_only);

  template void
  fill_all(tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy,
           const int                                 data_index,
           const int                                 coarsest_level_number,
           const int                                 finest_level_number,
           const double                              value,
           const bool                                interior_only);

  template void
  fill_all(tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy,
           const int                                 data_index,
           const int                                 coarsest_level_number,
           const int                                 finest_level_number,
           const float                               value,
           const bool                                interior_only);

  template tbox::Pointer<math::HierarchyDataOpsReal<NDIM, double>>
  extract_hierarchy_data_ops(
    const tbox::Pointer<hier::Variable<NDIM>> p,
    tbox::Pointer<hier::PatchHierarchy<NDIM>> patch_hierarchy);
} // namespace fdl
