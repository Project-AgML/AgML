# Helios standard CMakeLists.txt file version 1.3
cmake_minimum_required(VERSION 3.15)
project(helios)

#-------- USER INPUTS ---------#

#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file
set( BASE_DIRECTORY "../.." )

#define the name of the executable to be created
set( EXECUTABLE_NAME "SyntheticRadiation" )

#provide name of source file(s) (separate multiple file names with semicolon)
set( SOURCE_FILES "generate.cpp" )

#specify which plug-ins to use (separate plug-in names with semicolon)
set( PLUGINS "radiation" )

#-------- MAIN CODE (Dont Modify) ---------#
# set( WSL TRUE)
set( ENV{PATH} "$ENV{PATH}:/usr/local/cuda/bin" )
message("PATH: $ENV{PATH}")
set( ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:/usr/lib/wsl/lib" )
message("LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
SET(CMAKE_CXX_COMPILER_ID "GNU")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g" )
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
set( EXECUTABLE_NAME_EXT ${EXECUTABLE_NAME}_exe )
set( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )
add_executable( ${EXECUTABLE_NAME_EXT} ${SOURCE_FILES} )
add_subdirectory( ${BASE_DIRECTORY}/core "lib" )
target_link_libraries( ${EXECUTABLE_NAME_EXT} ${CMAKE_DL_LIBS} helios)
LIST(LENGTH PLUGINS PLUGIN_COUNT)
message("-- Loading ${PLUGIN_COUNT} plug-ins")
foreach(PLUGIN ${PLUGINS})
	message("-- loading plug-in ${PLUGIN}")
	add_subdirectory( ${BASE_DIRECTORY}/plugins/${PLUGIN} "plugins/${PLUGIN}" )
	target_link_libraries( ${EXECUTABLE_NAME_EXT} ${PLUGIN} )
endforeach(PLUGIN)
include_directories( "${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIRECTORY}" )
add_custom_command( TARGET ${EXECUTABLE_NAME_EXT} POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${EXECUTABLE_NAME_EXT} ${EXECUTABLE_NAME} )