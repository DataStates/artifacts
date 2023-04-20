cmake_minimum_required (VERSION 3.12)
project (dummy CXX)

set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build." FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")
set (TMCI_INCLUDE_DIR $ENV{HOME}/.local/include/python3.6m CACHE STRING "Path to TMCI headers")
set (TENSORFLOW_INCLUDE_DIR /soft/datascience/tensorflow/tf2.0-py36/tensorflow_core/include CACHE STRING "Path to Tensorflow headers")

find_package (Python REQUIRED)
find_package (PythonLibs REQUIRED)

# Find TMCI
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library_dir())"
	OUTPUT_VARIABLE TMCI_LIBRARY_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library())"
	OUTPUT_VARIABLE TMCI_LIBRARIES
	OUTPUT_STRIP_TRAILING_WHITESPACE)
message (STATUS "TMCI library dir: ${TMCI_LIBRARY_DIR}")
message (STATUS "TMCI library: ${TMCI_LIBRARIES}")

link_directories (${TMCI_LIBRARY_DIR})

include_directories (${PYTHON_INCLUDE_DIR})
include_directories (${TMCI_INCLUDE_DIR})
include_directories (${TENSORFLOW_INCLUDE_DIR})

add_subdirectory (src)
