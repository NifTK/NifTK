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
# Camino - external project for diffusion imaging.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED camino_DIR AND NOT EXISTS ${camino_DIR})
  message(FATAL_ERROR "camino_DIR variable is defined but corresponds to non-existing directory \"${camino_DIR}\".")
endif()

if(BUILD_CAMINO AND NOT WIN32)

  find_package(Java COMPONENTS Development)

  if(NOT "${Java_VERSION}" STREQUAL "")

    niftkMacroGetCommitHashOfCurrentFile(config_version)

    set(proj camino)
    set(proj_VERSION ${NIFTK_VERSION_CAMINO})
    set(proj_SOURCE ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-src)
    set(proj_CONFIG ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-cmake)
    set(proj_DEPENDENCIES)
    set(camino_DEPENDS ${proj})

    if(NOT DEFINED camino_DIR)

      niftkMacroGetChecksum(NIFTK_CHECKSUM_CAMINO ${NIFTK_LOCATION_CAMINO})

      ExternalProject_Add(${proj}
        SOURCE_DIR ${proj_SOURCE}
        PREFIX ${proj_CONFIG}
        URL ${NIFTK_LOCATION_CAMINO}
        URL_MD5 ${NIFTK_CHECKSUM_CAMINO}
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE ON
        LOG_BUILD ON
        CMAKE_ARGS
          ${EP_COMMON_ARGS}
        DEPENDS ${proj_DEPENDENCIES}
      )

      set(camino_DIR ${proj_SOURCE})
      message("SuperBuild loading Camino from ${camino_DIR}")

    else(NOT DEFINED camino_DIR)

      mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

    endif(NOT DEFINED camino_DIR)

  endif()

endif()
