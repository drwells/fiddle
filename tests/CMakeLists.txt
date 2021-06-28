ADD_CUSTOM_TARGET(tests)

SET(TEST_DIRECTORIES base grid interaction mechanics)

FOREACH(_dir ${TEST_DIRECTORIES})
  ADD_CUSTOM_TARGET("tests-${_dir}")
  ADD_DEPENDENCIES(tests "tests-${_dir}")
ENDFOREACH()

MACRO(SETUP _dir _src _target_link)
  GET_FILENAME_COMPONENT(_out_name "${_src}" NAME_WE)
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE "${_target_link}")
  ADD_DEPENDENCIES("tests-${_dir}" ${_target})
ENDMACRO()

MACRO(SETUP_2D _dir _src)
  GET_FILENAME_COMPONENT(_dest "${_src}" NAME_WE)
  SET(_out_name "${_dest}_2d")
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE fiddle2d)
  ADD_DEPENDENCIES("tests-${_dir}" ${_target})
ENDMACRO()

MACRO(SETUP_3D _dir _src)
  GET_FILENAME_COMPONENT(_dest "${_src}" NAME_WE)
  SET(_out_name "${_dest}_3d")
  SET(_target "tests-${_dir}_${_out_name}")
  ADD_EXECUTABLE(${_target} EXCLUDE_FROM_ALL "${_dir}/${_src}")
  SET_TARGET_PROPERTIES(${_target}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/tests/${_dir}"
    OUTPUT_NAME
    ${_out_name}
    )
  TARGET_COMPILE_OPTIONS(${_target} PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}/tests/${_dir}/")
  TARGET_LINK_LIBRARIES(${_target} PRIVATE fiddle3d)
  ADD_DEPENDENCIES("tests-${_dir}" ${_target})
ENDMACRO()

# test definitions:

# base:
SETUP(base hello.cc fiddle2d)
SETUP(base qgauss_family_01.cc fiddle3d)
SETUP(base qgauss_family_02.cc fiddle3d)

SETUP(base copy_database.cc fiddle2d)

SETUP_2D(base nonintersecting_sphere_01.cc)
SETUP_3D(base nonintersecting_sphere_01.cc)

# grid:
SETUP_2D(grid collect_bboxes_01.cc)
SETUP_3D(grid collect_bboxes_01.cc)

SETUP(grid fe_predicate_01.cc fiddle2d)
SETUP(grid grid_predicate_01.cc fiddle2d)
SETUP(grid overlap_tria_01.cc fiddle2d)
SETUP(grid patch_map_01.cc fiddle2d)
SETUP(grid patch_map_02.cc fiddle2d)

SETUP(grid tag_cells_01.cc fiddle2d)

# interaction:
SETUP(interaction count_quadrature_points_01.cc fiddle2d)

SETUP_2D(interaction ifed_tag.cc)
SETUP_3D(interaction ifed_tag.cc)

SETUP_2D(interaction ifed_interpolate_01.cc)
SETUP_2D(interaction ifed_spread_01.cc)

SETUP_2D(interaction ifed_ex4.cc)

SETUP_2D(interaction elemental_interpolate_01.cc)

SETUP(interaction interpolate_01.cc fiddle2d)
SETUP(interaction interpolate_02.cc fiddle3d)

SETUP(interaction spread_01.cc fiddle2d)

SETUP(interaction interaction_base_01.cc fiddle2d)

# mechanics:
SETUP(mechanics me_values_01.cc fiddle2d)

SETUP(mechanics pk1_volumetric_01.cc fiddle2d)
SETUP(mechanics pk1_volumetric_02.cc fiddle2d)
SETUP(mechanics pk1_volumetric_03.cc fiddle2d)

ADD_CUSTOM_COMMAND(TARGET tests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/attest ${CMAKE_BINARY_DIR}/attest)

FOREACH(_dir ${TEST_DIRECTORIES})
  ADD_CUSTOM_COMMAND(TARGET "tests-${_dir}"
    POST_BUILD
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/link-test-files.sh
    ${CMAKE_SOURCE_DIR}/tests/${_dir} ${CMAKE_BINARY_DIR}/tests/${_dir}
    VERBATIM)
ENDFOREACH()

# Find numdiff, if possible (we only need it for tests so its not essential that
# we find it now)
FIND_PROGRAM(NUMDIFF_EXECUTABLE NAMES numdiff HINTS ${NUMDIFF_ROOT} PATH_SUFFIXES bin)

IF ("${NUMDIFF_EXECUTABLE}" STREQUAL "NUMDIFF_EXECUTABLE-NOTFOUND")
  MESSAGE(WARNING "\
The configuration script was not able to locate numdiff. If you want to run \
the test suite you will need to either edit attest.conf, specify the path to \
numdiff to attest, or rerun CMake with the argument NUMDIFF_ROOT specifying \
numdiff's root installation directory.")
  # clear the value so that attest.conf doesn't contain an invalid path
  SET(NUMDIFF_EXECUTABLE "")
ENDIF()

# Set up the default attest configuration file:
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/tests/attest.conf.in
  ${CMAKE_BINARY_DIR}/attest.conf)
