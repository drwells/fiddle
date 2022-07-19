#ifndef included_fiddle_base_samrai_utilities_h
#define included_fiddle_base_samrai_utilities_h

#include <fiddle/base/config.h>

#include <BasePatchLevel.h>
#include <HierarchyDataOpsReal.h>
#include <PatchData.h>
#include <PatchHierarchy.h>
#include <PatchLevel.h>
#include <Variable.h>
#include <tbox/Database.h>
#include <tbox/MemoryDatabase.h>

#include <utility>
#include <vector>

// Collect the various hacks needed to work around common problems in SAMRAI.

namespace fdl
{
  using namespace SAMRAI;

  /**
   * Helper function for extracting locally owned patches from a base patch
   * level.
   */
  template <int spacedim>
  std::vector<tbox::Pointer<hier::Patch<spacedim>>>
  extract_patches(
    tbox::Pointer<hier::BasePatchLevel<spacedim>> base_patch_level);

  /**
   * Helper function for extracting locally owned patches from a patch level.
   */
  template <int spacedim>
  std::vector<tbox::Pointer<hier::Patch<spacedim>>>
  extract_patches(tbox::Pointer<hier::PatchLevel<spacedim>> patch_level);

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
  extract_types(const tbox::Pointer<hier::PatchData<spacedim>> &p);

  /**
   * Each class inheriting from PatchData implements getDepth() but we need to
   * downcast to call it.
   */
  template <int spacedim>
  int
  extract_depth(const tbox::Pointer<hier::PatchData<spacedim>> &p);

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
           const field_type                         value = 0);


  /**
   * Same as above, but for a patch hierarchy and data index.
   */
  template <int spacedim, typename field_type = int>
  void
  fill_all(tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy,
           const int                                     data_index,
           const int                                     coarsest_level_number,
           const int                                     finest_level_number,
           const field_type                              value         = 0,
           const bool                                    interior_only = false);

  /**
   * Like elsewhere, SAMRAI doesn't provide any way to actually subtract two
   * sets of data in a generic way, so we need to implement our own lookup code.
   */
  template <int spacedim>
  tbox::Pointer<math::HierarchyDataOpsReal<spacedim, double>>
  extract_hierarchy_data_ops(
    const tbox::Pointer<hier::Variable<spacedim>> p,
    tbox::Pointer<hier::PatchHierarchy<spacedim>> patch_hierarchy);

  /**
   * Copy the contents of the database into a new database.
   */
  tbox::Pointer<tbox::MemoryDatabase>
  copy_database(const tbox::Pointer<tbox::MemoryDatabase> &input,
                const std::string name_suffix = "::clone");

  /**
   * Save the binary representation of an object in a database.
   *
   * @note This function does an additional translation to base64 to avoid a bug
   * in SAMRAI.
   */
  void
  save_binary(const std::string &            key,
              const char *                   begin,
              const char *                   end,
              tbox::Pointer<tbox::Database> &database);

  /**
   * Load the binary representation of an object from a database. This is the
   * inverse of save_binary.
   *
   * @note This function does an additional translation from base64 to avoid a
   * bug in SAMRAI.
   */
  std::string
  load_binary(const std::string &                  key,
              const tbox::Pointer<tbox::Database> &database);
} // namespace fdl

#endif
