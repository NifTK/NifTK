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

  set(proj NiftyLink)
  set(proj_DEPENDENCIES)
  set(NIFTYLINK_DEPENDS ${proj})

  if(NOT DEFINED NiftyLink_DIR)

    set(revision_tag development)

    if (NIFTK_NIFTYLINK_DEV)
      set(NiftyLink_location_options
        GIT_REPOSITORY ${NIFTK_LOCATION_NIFTYLINK_REPOSITORY}
        GIT_TAG ${revision_tag}
      )
    else ()
      niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYLINK ${NIFTK_LOCATION_NIFTYLINK_TARBALL})
      set(NiftyLink_location_options
        URL ${NIFTK_LOCATION_NIFTYLINK_TARBALL}
        URL_MD5 ${NIFTK_CHECKSUM_NIFTYLINK}
      )
    endif ()

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
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      ${NiftyLink_location_options}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_NIFTYLINK}
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

    set(NiftyLink_DIR ${CMAKE_BINARY_DIR}/${proj}-build/NiftyLink-build)
    set(NiftyLink_SOURCE_DIR ${CMAKE_BINARY_DIR}/NiftyLink-src)
    set(OpenIGTLink_DIR ${CMAKE_BINARY_DIR}/${proj}-build/OPENIGTLINK-build)

    message("SuperBuild loading NiftyLink from ${NiftyLink_DIR}")
    message("SuperBuild loading OpenIGTLink from ${OpenIGTLink_DIR}")

  else(NOT DEFINED NiftyLink_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NiftyLink_DIR)

endif(BUILD_IGI)
