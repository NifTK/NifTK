#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND MSVC ) 

  set(_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMake/CMakeExternals/curl_patch.cmake" )
  message("\n ********* Adding patch to: ${_PATCH_FILE} ********* \n" )  

  niftkMacroGetChecksum(NIFTK_CHECKSUM_curl ${NIFTK_LOCATION_curl})

  ExternalProject_Add(${proj}
    BINARY_DIR ${proj}-build
    URL ${NIFTK_LOCATION_curl}
    URL_MD5 ${NIFTK_CHECKSUM_curl}
    PATCH_COMMAND "${CMAKE_COMMAND};-P;${_PATCH_FILE}"
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
else()  
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
endif()  



else(NOT DEFINED curl_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED curl_DIR)
