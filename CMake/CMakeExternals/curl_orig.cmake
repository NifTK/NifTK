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

#-----------------------------------------------------------------------------
# curl
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED curl_DIR AND NOT EXISTS ${curl_DIR})
  message(FATAL_ERROR "curl_DIR variable is defined but corresponds to non-existing directory.")
endif()

set(proj curl)
set(proj_DEPENDENCIES)
set(curl_DEPENDS ${proj})

if(NOT DEFINED curl_DIR)

  set(curl_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/curl-build)
  set(curl_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeExternals/Source/curl)
  
  message("SuperBuild loading curl from ${curl_DIR}")

  ExternalProject_Add(${proj}
    BINARY_DIR ${proj}-build
    URL ${NIFTK_LOCATION_curl}
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
      -DBUILD_CURL_EXE:BOOL=OFF
      -DBUILD_CURL_TESTS:BOOL=OFF
      -DCURL_STATICLIB:BOOL=OFF
    DEPENDS ${proj_DEPENDENCIES}
  )

else(NOT DEFINED curl_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED curl_DIR)
