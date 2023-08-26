CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

SET(FDL_VERSION_MAJOR 1)
SET(FDL_VERSION_MINOR 0)
SET(FDL_VERSION_PATCH 0)

SET(FIDDLE_VERSION "${FDL_VERSION_MAJOR}.${FDL_VERSION_MINOR}.${FDL_VERSION_PATCH}")
PROJECT(FIDDLE
  DESCRIPTION "Four-chambered heart + IBAMR + deal.II"
  VERSION ${FIDDLE_VERSION}
  # include C so that we can link against C libraries (e.g., MPI::MPI_C) easily
  LANGUAGES C CXX)

INCLUDE(CMakePackageConfigHelpers)
INCLUDE(GNUInstallDirs)

#
# Options for configuring fiddle itself
#

# Do we want dynamic or static linking?
OPTION(BUILD_SHARED_LIBS "Whether or not to build shared libraries." ON)

# Do we want additional barriers to make timers more accurate?
OPTION(FDL_ENABLE_TIMER_BARRIERS
  "Whether or not to add barriers before running top-level timers to improve their accuracy."
  ON)

OPTION(FDL_IGNORE_DEPENDENCY_FLAGS
"Whether or not to unset all flags set by CMake and deal.II (but not IBAMR's \
NDIM definition) and solely rely on CMAKE_CXX_FLAGS. Defaults to OFF. This \
option is useful if you want very precise control over which flags are actually \
used by CMake to compile fiddle - in most cases the flags set by deal.II are \
correct." OFF)

# configure RPATH:
SET(CMAKE_MACOSX_RPATH 1)
SET(CMAKE_BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_MACOSX_RPATH TRUE)

#
# set up dependencies:
#
FIND_PACKAGE(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_ROOT} $ENV{DEAL_II_DIR} $ENV{DEAL_II_ROOT})
IF(NOT ${DEAL_II_WITH_MPI})
  MESSAGE(FATAL_ERROR "fiddle requires that deal.II be compiled with support for MPI.")
ENDIF()

FIND_PACKAGE(IBAMR 0.11.0 REQUIRED HINTS ${IBAMR_ROOT} $ENV{IBAMR_ROOT})

#
# Modify CMake and dependencies if requested:
#
IF(${FDL_IGNORE_DEPENDENCY_FLAGS})
  MESSAGE(STATUS "Clearing deal.II targets' flags")
  FOREACH(_suffix "" "_debug" "_release")
    IF(TARGET "dealii::dealii${_suffix}")
      SET_TARGET_PROPERTIES("dealii::dealii${_suffix}"
        PROPERTIES
        INTERFACE_COMPILE_OPTIONS ""
        INTERFACE_COMPILE_DEFINITIONS "")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_CXX_FLAGS_DEBUG "")
  SET(CMAKE_CXX_FLAGS_RELEASE "")
ENDIF()

#
# set up the library:
#
SET(FIDDLE_DIMENSIONS "2" "3")

SET(FIDDLE_SRC
  source/base/samrai_utilities.cc
  source/base/quadrature_family.cc
  source/base/utilities.cc
  source/base/initial_guess.cc

  source/grid/box_utilities.cc
  source/grid/data_in.cc
  source/grid/grid_utilities.cc
  source/grid/intersection_predicate.cc
  source/grid/nodal_patch_map.cc
  source/grid/overlap_tria.cc
  source/grid/patch_map.cc
  source/grid/surface_tria.cc
  source/grid/triangle.c

  source/interaction/dlm_method.cc
  source/interaction/elemental_interaction.cc
  source/interaction/ifed_method.cc
  source/interaction/ifed_method_base.cc
  source/interaction/interaction_base.cc
  source/interaction/interaction_utilities.cc
  source/interaction/nodal_interaction.cc

  source/mechanics/mechanics_utilities.cc
  source/mechanics/mechanics_values.cc
  source/mechanics/force_contribution_lib.cc
  source/mechanics/part.cc
  source/mechanics/part_vectors.cc
  source/mechanics/fiber_network.cc

  source/postprocess/point_values.cc
  source/postprocess/surface_meter.cc


  source/transfer/overlap_partitioning_tools.cc
  source/transfer/scatter.cc
  )
# Always compile triangle as C++
SET_SOURCE_FILES_PROPERTIES(${FIDDLE_SRC} PROPERTIES LANGUAGE CXX)

# set up the configuration header:
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/config.h.in
  ${CMAKE_BINARY_DIR}/include/fiddle/base/config.h)
INSTALL(FILES ${CMAKE_BINARY_DIR}/include/fiddle/base/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fiddle/base/)

# like IBAMR, we compile in 2D and 3D but only install the headers once.
ADD_LIBRARY(FIDDLEHeaders INTERFACE)
TARGET_INCLUDE_DIRECTORIES(
  FIDDLEHeaders
  INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)
INSTALL(TARGETS FIDDLEHeaders EXPORT FIDDLETargets)
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fiddle
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# set up targets for export:
SET(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/fiddle/)

# Set up actual files containing the export target information:
INSTALL(EXPORT "FIDDLETargets" FILE "FIDDLETargets.cmake"
  NAMESPACE FIDDLE::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fiddle/)

# Do the same for in-tree:
EXPORT(EXPORT "FIDDLETargets"
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/FIDDLETargets.cmake"
  NAMESPACE FIDDLE::)

WRITE_BASIC_PACKAGE_VERSION_FILE(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/FIDDLEConfigVersion.cmake"
  VERSION ${FIDDLE_VERSION}
  COMPATIBILITY AnyNewerVersion)

CONFIGURE_PACKAGE_CONFIG_FILE(
  ${CMAKE_SOURCE_DIR}/cmake/FIDDLEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/FIDDLEConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/FIDDLEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/FIDDLEConfigVersion.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

# set up the libraries themselves:
FOREACH(_d ${FIDDLE_DIMENSIONS})
  SET(_lib "fiddle${_d}d")
  ADD_LIBRARY(${_lib})
  TARGET_SOURCES(${_lib} PRIVATE ${FIDDLE_SRC})

  TARGET_LINK_LIBRARIES(${_lib} PUBLIC FIDDLEHeaders)

  TARGET_INCLUDE_DIRECTORIES(${_lib} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)

  # add some flags to get triangle compiling
  TARGET_COMPILE_OPTIONS(${_lib} PRIVATE -DANSI_DECLARATORS)
  TARGET_COMPILE_OPTIONS(${_lib} PRIVATE -DTRILIBRARY)

  # and dependencies
  TARGET_LINK_LIBRARIES(${_lib} PUBLIC dealii::dealii)
  TARGET_LINK_LIBRARIES(${_lib} PUBLIC "IBAMR::IBAMR${_d}d")

  INSTALL(TARGETS ${_lib} EXPORT FIDDLETargets COMPONENT library)
ENDFOREACH()

ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(examples)

#
# Provide "indent" target for indenting all headers and source files
#
add_custom_target(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./scripts/indent-all
  COMMENT "Indenting all fiddle files"
  )
