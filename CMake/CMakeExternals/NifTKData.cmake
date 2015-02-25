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

  set(version "b7c346650c")
  set(location "https://cmiclab.cs.ucl.ac.uk/CMIC/NifTKData.git")

  niftkMacroDefineExternalProjectVariables(NifTKData ${version} ${location})

  if (NOT DEFINED NIFTK_DATA_DIR)

    ExternalProject_Add(${proj}
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      GIT_REPOSITORY ${proj_LOCATION}
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
