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
if(DEFINED SlicerExecutionModel_DIR AND NOT EXISTS ${SlicerExecutionModel_DIR})
  message(FATAL_ERROR "SlicerExecutionModel_DIR variable is defined but corresponds to non-existing directory \"${SlicerExecutionModel_DIR}\".")
endif()

niftkMacroGetCommitHashOfCurrentFile(config_version)

set(proj SlicerExecutionModel)
set(proj_VERSION ${NIFTK_VERSION_SEM})
set(proj_SOURCE ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-src)
set(proj_CONFIG ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-cmake)
set(proj_BUILD ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-build)
set(proj_INSTALL ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-install)
set(proj_DEPENDENCIES ITK)
set(SlicerExecutionModel_DEPENDS ${proj})

if(NOT DEFINED SlicerExecutionModel_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_SEM ${NIFTK_LOCATION_SEM})

  ExternalProject_Add(${proj}
    SOURCE_DIR ${proj_SOURCE}
    PREFIX ${proj_CONFIG}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${NIFTK_LOCATION_SEM}
    URL_MD5 ${NIFTK_CHECKSUM_SEM}
    UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      ${additional_cmake_args}
      -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
      -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
      -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
      -DSlicerExecutionModel_USE_JSONCPP:BOOL=OFF
      -DITK_DIR:PATH=${ITK_DIR}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(SlicerExecutionModel_DIR ${proj_BUILD})
  message("SuperBuild loading SlicerExecutionModel from ${SlicerExecutionModel_DIR}")

else(NOT DEFINED SlicerExecutionModel_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED SlicerExecutionModel_DIR)
