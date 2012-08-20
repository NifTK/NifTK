
set(zlib_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../CMakeExternals/Source/zlib)
list(APPEND ALL_INCLUDE_DIRECTORIES ${zlib_INCLUDE_DIR})
include_directories(${zlib_INCLUDE_DIR})

set(zlib_LIBRARY_DIR ${CMAKE_BINARY_DIR}/../zlib-build)
link_directories(${zlib_LIBRARY_DIR})
list(APPEND ALL_LIBRARY_DIRS ${zlib_LIBRARY_DIR})

list(APPEND ALL_LIBRARIES z)
