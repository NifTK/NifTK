#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(curl_INCLUDE_DIR ${CMAKE_BINARY_DIR}/../CMakeExternals/Source/curl/include/curl)
list(APPEND ALL_INCLUDE_DIRECTORIES ${curl_INCLUDE_DIR})
include_directories(${curl_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/../curl-build/include/curl)

set(curl_LIBRARY_DIR ${CMAKE_BINARY_DIR}/../curl-build/lib)
link_directories(${curl_LIBRARY_DIR})
list(APPEND ALL_LIBRARY_DIRS ${curl_LIBRARY_DIR})

if (WIN32)
  set(CURL_BINARY_DIR ${CMAKE_BINARY_DIR}/../curl-build/lib/${CMAKE_BUILD_TYPE})
  set(curl_LIBRARIES libcurl_imp)
else ()
  set(curl_LIBRARIES curl)
endif()

list(APPEND ALL_LIBRARIES ${curl_LIBRARIES})
