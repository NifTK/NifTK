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

    set(NIFTK_VERSION_Camino "4612bee5fa" CACHE STRING "Version of Camino" FORCE)
    set(NIFTK_LOCATION_Camino "${NIFTK_EP_TARBALL_LOCATION}/camino-${NIFTK_VERSION_Camino}.tar.gz" CACHE STRING "Location of Camino package" FORCE)

    niftkMacroDefineExternalProjectVariables(Camino ${NIFTK_VERSION_Camino})

    if(NOT DEFINED camino_DIR)

      niftkMacroGetChecksum(NIFTK_CHECKSUM_Camino ${NIFTK_LOCATION_Camino})

      ExternalProject_Add(${proj}
        SOURCE_DIR ${proj_SOURCE}
        PREFIX ${proj_CONFIG}
        URL ${NIFTK_LOCATION_Camino}
        URL_MD5 ${NIFTK_CHECKSUM_Camino}
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
