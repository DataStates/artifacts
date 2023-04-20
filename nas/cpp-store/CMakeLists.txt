cmake_minimum_required (VERSION 3.12)
project (dummy CXX)
set(CMAKE_CXX_STANDARD 17)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build." FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(thallium REQUIRED)
find_package(Python REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library_dir())"
	OUTPUT_VARIABLE TMCI_LIBRARY_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library())"
	OUTPUT_VARIABLE TMCI_LIBRARIES
	OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tensorflow as tf; print(*tf.sysconfig.get_compile_flags())"
  OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tensorflow as tf; print(*tf.sysconfig.get_link_flags())"
  OUTPUT_VARIABLE TENSORFLOW_LINK_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE)

message (STATUS "TMCI library dir: '${TMCI_LIBRARY_DIR}'")
message (STATUS "TMCI library: '${TMCI_LIBRARIES}'")
message (STATUS "Tensorflow compile_flags: '${TENSORFLOW_COMPILE_FLAGS}'")
message (STATUS "Tensorflow library: '${TENSORFLOW_LINK_FLAGS}'")

link_directories (${TMCI_LIBRARY_DIR})

set (DUMMY_SRC model_client.cpp DummyBackend.cpp model_server.cpp)
add_library (dummy ${DUMMY_SRC})

target_include_directories(dummy PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../venv/include/"
  ${TMCI_INCLUDE_DIR}
  )

target_link_libraries(dummy PUBLIC
  :${TMCI_LIBRARIES}
  CUDA::cuda_driver
  CUDA::cudart
  thallium
  )
target_link_options(dummy PUBLIC
  SHELL:${TENSORFLOW_LINK_FLAGS}
  )
target_compile_options(dummy PUBLIC
  SHELL:${TENSORFLOW_COMPILE_FLAGS}
  )
target_compile_features(dummy PRIVATE cxx_std_17)

add_executable(server server.cpp model_server.cpp)
target_link_libraries(server PUBLIC
  CUDA::cuda_driver
  CUDA::cudart
  )
target_compile_features(server PRIVATE cxx_std_17)
target_link_libraries(server PRIVATE thallium MPI::MPI_CXX)
target_compile_definitions(server PRIVATE -D__INFO -D__BENCHMARK -D__ASSERT)
