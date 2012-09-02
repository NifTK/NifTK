
set(curl_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../CMakeExternals/Source/curl/include/curl)
list(APPEND ALL_INCLUDE_DIRECTORIES ${curl_INCLUDE_DIR})
include_directories(${curl_INCLUDE_DIR})

set(curl_LIBRARY_DIR ${CMAKE_BINARY_DIR}/../curl-build/lib)
link_directories(${curl_LIBRARY_DIR})
list(APPEND ALL_LIBRARY_DIRS ${curl_LIBRARY_DIR})

if (WIN32)
  set(curl_LIBRARIES libcurl libcurl_imp)
else ()
  set(curl_LIBRARIES curl)
endif()

list(APPEND ALL_LIBRARIES ${curl_LIBRARIES})
