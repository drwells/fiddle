CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

# Do we want dynamic or static linking?
OPTION(BUILD_SHARED_LIBS "Whether or not to build shared libraries." ON)

SET(FIDDLE_DIMENSIONS "2" "3")

PROJECT(FIDDLE
  DESCRIPTION "Four-chambered heart + IBAMR + deal.II"
  VERSION 0.0.1
  # include C so that we can link against C libraries (e.g., MPI::MPI_C) easily
  LANGUAGES C CXX)

#
# set up dependencies:
#
FIND_PACKAGE(deal.II 9.3.0 REQUIRED HINTS ${DEAL_II_ROOT} $ENV{DEAL_II_DIR} $ENV{DEAL_II_ROOT})
FIND_PACKAGE(IBAMR 0.8.0 REQUIRED HINTS ${IBAMR_ROOT} $ENV{IBAMR_ROOT})

#
# set up the library:
#
SET(FIDDLE_SRC
  source/base/samrai_utilities.cc
  source/base/quadrature_family.cc

  source/grid/overlap_tria.cc
  source/grid/patch_map.cc
  source/grid/box_utilities.cc

  source/interaction/elemental_interaction.cc
  source/interaction/ifed_method.cc
  source/interaction/interaction_base.cc
  source/interaction/interaction_utilities.cc

  source/mechanics/part.cc
  source/mechanics/mechanics_utilities.cc

  source/transfer/overlap_partitioning_tools.cc
  source/transfer/scatter.cc
  )

FOREACH(_d ${FIDDLE_DIMENSIONS})
  SET(_lib "fiddle${_d}d")
  ADD_LIBRARY(${_lib})
  TARGET_SOURCES(${_lib} PRIVATE ${FIDDLE_SRC})
  TARGET_INCLUDE_DIRECTORIES(${_lib} PUBLIC
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/>)

  # TODO - support multiple build types
  TARGET_LINK_LIBRARIES(${_lib} PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
  TARGET_INCLUDE_DIRECTORIES(${_lib} PUBLIC ${DEAL_II_INCLUDE_DIRS})

  TARGET_LINK_LIBRARIES(${_lib} PUBLIC "IBAMR::IBAMR${_d}d")
ENDFOREACH()

ADD_SUBDIRECTORY(tests)
