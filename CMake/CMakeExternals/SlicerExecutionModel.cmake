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
# SlicerExecutionModel
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED SlicerExecutionModel_DIR AND NOT EXISTS ${SlicerExecutionModel_DIR})
  MESSAGE(FATAL_ERROR "SlicerExecutionModel_DIR variable is defined but corresponds to non-existing directory \"${SlicerExecutionModel_DIR}\".")
ENDIF()

SET(proj SlicerExecutionModel)
SET(proj_DEPENDENCIES ITK)
SET(SlicerExecutionModel_DEPENDS ${proj})

IF(NOT DEFINED SlicerExecutionModel_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_SEM ${NIFTK_LOCATION_SEM})

  ExternalProject_Add(${proj}
     URL ${NIFTK_LOCATION_SEM}
     URL_MD5 ${NIFTK_CHECKSUM_SEM}
     BINARY_DIR ${proj}-build
     UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_SEM}
     INSTALL_COMMAND ""
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       ${additional_cmake_args}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DITK_DIR:PATH=${ITK_DIR}
     DEPENDS ${proj_DEPENDENCIES}
  )
 
  SET(SlicerExecutionModel_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading SlicerExecutionModel from ${SlicerExecutionModel_DIR}")

ELSE(NOT DEFINED SlicerExecutionModel_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED SlicerExecutionModel_DIR)
