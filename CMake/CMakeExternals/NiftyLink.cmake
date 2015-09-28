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
# NiftyLink
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NiftyLink_DIR AND NOT EXISTS ${NiftyLink_DIR})
  message(FATAL_ERROR "NiftyLink_DIR variable is defined but corresponds to non-existing directory \"${NIFTYLINK_DIR}\".")
endif()

if(BUILD_IGI)

  set(location "https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink.git")

  if (NIFTK_NIFTYLINK_DEV)

    # This retrieves the latest commit hash on the development branch.

    execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --heads ${location} development
       ERROR_VARIABLE GIT_error
       OUTPUT_VARIABLE NiftyLinkVersion
       OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT ${GIT_error} EQUAL 0)
      message(SEND_ERROR "Command \"${GIT_EXECUTABLE} ls-remote --heads ${location} development\" failed with output:\n${GIT_error}")
    endif()

    string(SUBSTRING ${NiftyLinkVersion} 0 10 version)

  else ()
    set(version "d733335df1")
  endif ()

  niftkMacroDefineExternalProjectVariables(NiftyLink ${version} ${location})

  if(NOT DEFINED NiftyLink_DIR)

    if(DEFINED NIFTYLINK_OIGTLINK_DEV)
      set(NiftyLink_options
        -DNIFTYLINK_OIGTLINK_DEV:BOOL=${NIFTYLINK_OIGTLINK_DEV}
      )
    else()
      set(NiftyLink_options
        -DNIFTYLINK_OIGTLINK_DEV:BOOL=${NIFTK_NIFTYLINK_DEV}
      )
    endif()

    if(NIFTYLINK_OPENIGTLINK_VERSION)
      list(APPEND NiftyLink_options -DNIFTYLINK_OPENIGTLINK_VERSION=${NIFTYLINK_OPENIGTLINK_VERSION} )
    endif()
    if(NIFTYLINK_OPENIGTLINK_MD5)
      list(APPEND NiftyLink_options -DNIFTYLINK_OPENIGTLINK_MD5=${NIFTYLINK_OPENIGTLINK_MD5} )
    endif()
    if(NIFTYLINK_OPENIGTLINK_LOCATION)
      list(APPEND NiftyLink_options -DNIFTYLINK_OPENIGTLINK_LOCATION=${NIFTYLINK_OPENIGTLINK_LOCATION} )
    endif()
    if(NIFTYLINK_OPENIGTLINK_LOCATION_DEV)
      list(APPEND NiftyLink_options -DNIFTYLINK_OPENIGTLINK_LOCATION_DEV=${NIFTYLINK_OPENIGTLINK_LOCATION_DEV} )
    endif()

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${proj_LOCATION}
      GIT_TAG ${proj_VERSION}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        ${NiftyLink_options}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NiftyLink_DIR ${proj_BUILD}/NiftyLink-build)
    set(NiftyLink_SOURCE_DIR ${proj_SOURCE})
    set(OpenIGTLink_DIR ${proj_BUILD}/OPENIGTLINK-build)

#    set(NiftyLink_DIR ${proj_INSTALL})
#    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
#    mitkFunctionInstallExternalCMakeProject(${proj})
    message("SuperBuild loading NiftyLink from ${NiftyLink_DIR}")
    message("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")

  else(NOT DEFINED NiftyLink_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NiftyLink_DIR)

endif(BUILD_IGI)
