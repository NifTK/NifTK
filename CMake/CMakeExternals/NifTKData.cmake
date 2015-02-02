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
# NifTKData - Downloads the unit-testing data as a separate project.
#-----------------------------------------------------------------------------

# Sanity checks
if (DEFINED NIFTK_DATA_DIR AND NOT EXISTS ${NIFTK_DATA_DIR})
  message(FATAL_ERROR "NIFTK_DATA_DIR variable is defined but corresponds to non-existing directory \"${NIFTK_DATA_DIR}\".")
endif ()

if (BUILD_TESTING)

  niftkMacroDefineExternalProjectVariables(NifTKData ${NIFTK_VERSION_NifTKData})

  if (NOT DEFINED NIFTK_DATA_DIR)

    set(${proj}_location ${NIFTK_LOCATION_DATA_GIT})
    set(${proj}_location_options
    )

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      GIT_REPOSITORY ${NIFTK_LOCATION_${proj}}
      GIT_TAG ${proj_VERSION}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NIFTK_DATA_DIR ${proj_SOURCE})
    message("SuperBuild loading ${proj} from ${NIFTK_DATA_DIR}")

  else ()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif (NOT DEFINED NIFTK_DATA_DIR)

endif (BUILD_TESTING)
