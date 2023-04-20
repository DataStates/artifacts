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
set (TMCI_INCLUDE_DIR "/home/mmadhya1/.local/lib/python3.8" CACHE STRING "Path to TMCI headers")
#set (TMCI_INCLUDE_DIR "/gpfs/mira-home/mmadhya1/dh1/lib/python3.8/site-packages/" CACHE STRING "Path to TMCI headers")
set (TENSORFLOW_INCLUDE_DIR "/lus/swift/home/mmadhya1/dh-tmcitest4/lib/python3.8/site-packages/tensorflow/include" CACHE STRING "Path to Tensorflow headers")
set(Python_ROOT_DIR ${Python_ROOT_DIR} )
#find_package (Python ${MY_PYTHON_VERSION} EXACT REQUIRED)
#find_package (PythonLibs ${MY_PYTHON_VERSION} EXACT REQUIRED)
#find_package (Python 3.7 REQUIRED)
#find_package (PythonLibs 3.7 REQUIRED)

set(Python_EXECUTABLE "/lus/swift/home/mmadhya1/dh-tmcitest4/bin/python3.8")
set(PYTHON_INCLUDE_DIR "/lus/swift/home/mmadhya1/dh-tmcitest4/include/python3.8")

# Find TMCI
message("python executable: ", ${Python_EXECUTABLE})
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
message("python include dir", ${PYTHON_INCLUDE_DIR})


include_directories (${TMCI_INCLUDE_DIR})
include_directories (${TENSORFLOW_INCLUDE_DIR})

add_subdirectory (src)
