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

set(version "11ea15b81e")
set(location "${NIFTK_EP_TARBALL_LOCATION}/Slicer-SlicerExecutionModel-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(SlicerExecutionModel ${version} ${location})
set(proj_DEPENDENCIES ITK)

if(NOT DEFINED SlicerExecutionModel_DIR)

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    PREFIX ${proj_CONFIG}
    SOURCE_DIR ${proj_SOURCE}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${proj_LOCATION}
    URL_MD5 ${proj_CHECKSUM}
    UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      ${additional_cmake_args}
      -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
      -DSlicerExecutionModel_USE_JSONCPP:BOOL=OFF
      -DITK_DIR:PATH=${ITK_DIR}
    DEPENDS ${proj_DEPENDENCIES}
  )

  if(EP_ALWAYS_USE_INSTALL_DIR)
    set(SlicerExecutionModel_DIR ${proj_INSTALL})
    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
  else()
    set(SlicerExecutionModel_DIR ${proj_BUILD})
  endif()

  message("SuperBuild loading SlicerExecutionModel from ${SlicerExecutionModel_DIR}")

else(NOT DEFINED SlicerExecutionModel_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED SlicerExecutionModel_DIR)
