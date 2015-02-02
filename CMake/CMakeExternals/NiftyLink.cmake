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

  if (NIFTK_NIFTYLINK_DEV)

    # This retrieves the latest commit hash on the development branch.

    execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --heads ${NIFTK_LOCATION_NiftyLink} development
       ERROR_VARIABLE GIT_error
       OUTPUT_VARIABLE NiftyLinkVersion
       OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT ${GIT_error} EQUAL 0)
      message(SEND_ERROR "Command \"${GIT_EXECUTABLE} ls-remote --heads ${NIFTK_LOCATION_NiftyLink} development\" failed with output:\n${GIT_error}")
    endif()

    string(SUBSTRING ${NiftyLinkVersion} 0 10 NiftyLinkVersion)

    set(NIFTK_VERSION_NiftyLink ${NiftyLinkVersion} CACHE STRING "Version of NiftyLink" FORCE)

  else ()

    set(NIFTK_VERSION_NiftyLink "b9f2782f73" CACHE STRING "Version of NiftyLink" FORCE)

  endif ()

  set(NIFTK_LOCATION_NiftyLink "https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink.git" CACHE STRING "Location of NiftyLink repository" FORCE)

  niftkMacroDefineExternalProjectVariables(NiftyLink ${NIFTK_VERSION_NiftyLink})

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
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${NIFTK_LOCATION_NiftyLink}
      GIT_TAG ${proj_VERSION}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${NiftyLink_options}
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NiftyLink_DIR ${proj_BUILD}/NiftyLink-build)
    set(NiftyLink_SOURCE_DIR ${proj_SOURCE})
    set(OpenIGTLink_DIR ${proj_BUILD}/OPENIGTLINK-build)

    message("SuperBuild loading NiftyLink from ${NiftyLink_DIR}")
    message("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")

  else(NOT DEFINED NiftyLink_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NiftyLink_DIR)

endif(BUILD_IGI)
