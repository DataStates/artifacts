set (DUMMY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/DummyBackend.cpp)

add_library (dummy ${DUMMY_SRC})
target_link_libraries (dummy ":${TMCI_LIBRARIES}" ) # we need the ":" because the library doesn't start with "lib"
