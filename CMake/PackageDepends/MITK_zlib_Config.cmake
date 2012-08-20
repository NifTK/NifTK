
set(zlib_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../CMakeExternals/Source/zlib)
set(zlib_LIBRARY_DIR ${CMAKE_BINARY_DIR}/../zlib-build)

list(APPEND ALL_INCLUDE_DIRECTORIES ${zlib_INCLUDE_DIR})
list(APPEND ALL_INCLUDE_DIRECTORIES ${zlib_LIBRARY_DIR})
include_directories(${zlib_INCLUDE_DIR})

link_directories(${zlib_LIBRARY_DIR})
list(APPEND ALL_LIBRARY_DIRS ${zlib_LIBRARY_DIR})

list(APPEND ALL_LIBRARIES z)
