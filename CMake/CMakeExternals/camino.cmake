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
# camino - external project for diffusion imaging.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED camino_DIR AND NOT EXISTS ${camino_DIR})
  message(FATAL_ERROR "camino_DIR variable is defined but corresponds to non-existing directory \"${camino_DIR}\".")
endif()

if(NOT WIN32 AND BUILD_CAMINO)

  find_package(Java COMPONENTS Development)
  find_package(Subversion)

  if(NOT "${Java_VERSION}" STREQUAL "" AND Subversion_FOUND)

    set(proj camino)
    set(proj_DEPENDENCIES)
    set(camino_DEPENDS ${proj})
    set(proj_SRC ${CMAKE_BINARY_DIR}/${proj}-src)
  
    if(NOT DEFINED camino_DIR)
  
      niftkMacroGetChecksum(NIFTK_CHECKSUM_CAMINO ${NIFTK_LOCATION_CAMINO})
  
      ExternalProject_Add(${proj}
        SOURCE_DIR ${proj}-src
        PREFIX ${proj}-cmake
        INSTALL_DIR ${proj}-install
        URL ${NIFTK_LOCATION_CAMINO}
        URL_MD5 ${NIFTK_CHECKSUM_CAMINO}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND make
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE ON
        LOG_BUILD ON
        DEPENDS ${proj_DEPENDENCIES}
        )
 
      set(camino_DIR ${proj_SRC})
      message("SuperBuild loading camino from ${camino_DIR}")

    else(NOT DEFINED camino_DIR)
  
      mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
    endif(NOT DEFINED camino_DIR)

  endif()

endif()


