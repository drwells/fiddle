SET(TARGET "fiddle")
CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

SET(FIDDLE_DIMENSIONS "2" "3")

PROJECT(FIDDLE
  DESCRIPTION "Four-chambered heart + IBAMR + deal.II"
  VERSION 0.0.1
  # include C so that we can link against C libraries (e.g., MPI::MPI_C) easily
  LANGUAGES C CXX)

#
# set up dependencies:
#
FIND_PACKAGE(deal.II 9.3.0 QUIET REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

#
# set up dependencies:
#
FIND_PACKAGE(deal.II 9.3.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
FIND_PACKAGE(IBAMR 0.8.0 REQUIRED HINTS ${IBAMR_DIR} $ENV{IBAMR_DIR})

#
# set up the library:
#
SET(FIDDLE_SRC ${TARGET}.cc
  source/grid/overlap_tria.cc
  source/grid/patch_map.cc
  source/grid/box_utilities.cc
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

  TARGET_LINK_LIBRARIES(${_lib} PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
  TARGET_INCLUDE_DIRECTORIES(${_lib} PUBLIC ${DEAL_II_INCLUDE_DIRS})

  TARGET_LINK_LIBRARIES(${_lib} PUBLIC "IBAMR::IBAMR${_d}d")
ENDFOREACH()

#
# setup the executable
#
FOREACH(_d ${FIDDLE_DIMENSIONS})
  SET(_exec "fiddle${_d}")
  ADD_EXECUTABLE(${_exec} "fiddle.cc")
  TARGET_LINK_LIBRARIES(${_exec} PUBLIC "fiddle${_d}d")
ENDFOREACH()

ADD_SUBDIRECTORY(tests)
