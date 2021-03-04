#ifndef included_fiddle_base_samrai_utilities_h
#define included_fiddle_base_samrai_utilities_h

#include <deal.II/base/exceptions.h>

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

#include <utility>

// Collect the various hacks needed to work around common problems in SAMRAI.

namespace fdl
{
  using namespace SAMRAI;

  /**
   * Several of SAMRAI's class hierarchies are poorly designed - in many
   * places you cannot access type-generic information without downcasting,
   * which requires knowledge of which class one should downcast to. For
   * example - every SAMRAI class inheriting from PatchData has a getDepth()
   * member function, but PatchData (despite introducing the concept of depth)
   * does not.
   *
   * This enum works around this mismatch between information needed at
   * compile time (the derived type, needed to access depth and other
   * attributes) and information available at run time. It can also be used to
   * template code.
   */
  enum class SAMRAIPatchType
  {
    Edge,
    Cell,
    Node,
    Side
  };

  /**
   * Similarly, SAMRAI doesn't propagate the type of the stored data up to
   * PatchData (even though every derived class has to implement this as a
   * template parameter), so convert it to an enum here:
   */
  enum class SAMRAIFieldType
  {
    Int,
    Float,
    Double
    // no support yet for complex
  };

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

    AssertThrow(false, dealii::ExcNotImplemented());
    return {};
  }


  /**
   * Each class inheriting from PatchData implements getDepth() but we need to
   * downcast to call it.
   */
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

    AssertThrow(false, dealii::ExcNotImplemented());
    return {};
  }


  /**
   * Like depth, each class inheriting from PatchData implements fillAll and is
   * templated on type, but none of this information is available without
   * explicitly downcasting.
   *
   * The input type defaults to integers since that can be reasonably cast to
   * floats and doubles.
   */
  template <int spacedim, typename field_type = int>
  void
  fill_all(tbox::Pointer<hier::PatchData<spacedim>> p,
           const field_type                         value = 0)
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

    AssertThrow(false, dealii::ExcNotImplemented());
  }

  /**
   * Like elsewhere, SAMRAI doesn't provide any way to actually subtract two
   * sets of data in a generic way, so we need to implement our own lookup code.
   */
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

    AssertThrow(false, dealii::ExcNotImplemented());
  }
} // namespace fdl

#endif
