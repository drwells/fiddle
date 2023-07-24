CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

SET(FIDDLE_VERSION 0.0.1)
PROJECT(FIDDLE
  DESCRIPTION "Four-chambered heart + IBAMR + deal.II"
  VERSION ${FIDDLE_VERSION}
  # include C so that we can link against C libraries (e.g., MPI::MPI_C) easily
  LANGUAGES C CXX)

INCLUDE(CMakePackageConfigHelpers)
INCLUDE(GNUInstallDirs)

# Do we want dynamic or static linking?
OPTION(BUILD_SHARED_LIBS "Whether or not to build shared libraries." ON)

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

FIND_PACKAGE(IBAMR 0.12.0 REQUIRED HINTS ${IBAMR_ROOT} $ENV{IBAMR_ROOT})

#
# Check that both IBAMR and deal.II use the same copies of common dependencies
#
FUNCTION(CHECK_DEPENDENCY dependency subdirectory dealii_interface_name
    ibamr_root_directory filename)
  MESSAGE(STATUS "")
  MESSAGE(STATUS
"Checking that IBAMR and deal.II use the same version of ${dependency}")
  GET_TARGET_PROPERTY(_dealii_include_directories
    "dealii::interface_${dealii_interface_name}"
    INTERFACE_INCLUDE_DIRECTORIES)

  # HINTS/PATHS doesn't work with semicolon-separated lists so do that here
  SET(_old_cmake_include_path ${CMAKE_INCLUDE_PATH})
  SET(CMAKE_INCLUDE_PATH ${_dealii_include_directories})
  FIND_FILE(_dealii_${dependency}_file "${filename}"
    PATH_SUFFIXES "${subdirectory}"
    REQUIRED)
  SET(CMAKE_INCLUDE_PATH ${_old_cmake_include_path})

  # Work around an IBAMR bug in which dependencies are not explicitly exported
  # when they are in system locations
  IF(NOT "${ibamr_root_directory}" STREQUAL "")
    SET(CMAKE_INCLUDE_PATH "${ibamr_root_directory}/include;${ibamr_root_directory}")
  ENDIF()
  FIND_FILE(_ibamr_${dependency}_file "${filename}"
    PATH_SUFFIXES "${subdirectory}"
    REQUIRED)

  MESSAGE(STATUS "")
  MESSAGE(STATUS "  deal.II ${dependency} file: ${_dealii_${dependency}_file}")
  MESSAGE(STATUS "  IBAMR   ${dependency} file: ${_ibamr_${dependency}_file}")

  IF(NOT "${_dealii_${dependency}_file}" STREQUAL "${_ibamr_${dependency}_file}")
    MESSAGE(FATAL_ERROR
      "The version of ${dependency} used by deal.II has headers in             \
${_dealii_include_directories} whereas the version of boost used by IBAMR      \
has headers in ${ibamr_root_directory}. Fiddle requires that both IBAMR and    \
deal.II use the same externally-installed copy of ${dependency}.")
  ENDIF()
  SET(CMAKE_INCLUDE_PATH ${_old_cmake_include_path})
ENDFUNCTION()

CHECK_DEPENDENCY(Boost "boost" boost "${Boost_ROOT}" version.hpp)
CHECK_DEPENDENCY(MPI "" mpi "${MPI_HOME}" mpi.h)

# TODO: verify that we use the same copy of PETSc in both cases. IBAMR does not
# presently export any information about PETSc so this cannot yet be done.

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
  # if we want to write a config header it should be added as a dependency here

  # Check for the new version
  IF("${DEAL_II_INCLUDE_DIRS}" STREQUAL "")
    SET(_use_dealii_targets TRUE)
  ELSE()
    SET(_use_dealii_targets FALSE)
  ENDIF()

  # add some flags to get triangle compiling
  TARGET_COMPILE_OPTIONS(${_lib} PRIVATE -DANSI_DECLARATORS)
  TARGET_COMPILE_OPTIONS(${_lib} PRIVATE -DTRILIBRARY)

  IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    IF(${_use_dealii_targets})
      TARGET_LINK_LIBRARIES(${_lib} PUBLIC dealii::dealii_debug)
    ELSE()
      TARGET_LINK_LIBRARIES(${_lib} PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
    ENDIF()
  ELSEIF(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE" OR ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    IF(${_use_dealii_targets})
      TARGET_LINK_LIBRARIES(${_lib} PUBLIC dealii::dealii_release)
    ELSE()
      TARGET_LINK_LIBRARIES(${_lib} PUBLIC ${DEAL_II_LIBRARIES_RELEASE})
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR "CMAKE_BUILD_TYPE should be DEBUG or RELEASE")
  ENDIF()

  IF(NOT $(_use_dealii_targets))
    TARGET_INCLUDE_DIRECTORIES(${_lib} PUBLIC ${DEAL_II_INCLUDE_DIRS})
  ENDIF()

  TARGET_LINK_LIBRARIES(${_lib} PUBLIC "IBAMR::IBAMR${_d}d")

  INSTALL(TARGETS ${_lib} EXPORT FIDDLETargets COMPONENT library)
ENDFOREACH()

ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(examples)
