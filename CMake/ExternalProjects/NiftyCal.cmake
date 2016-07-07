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
# NiftyCal
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NiftyCal_DIR AND NOT EXISTS ${NiftyCal_DIR})
  message(FATAL_ERROR "NiftyCal_DIR variable is defined but corresponds to non-existing directory \"${NiftyCal_DIR}\".")
endif()

if(BUILD_IGI)

  set(location "https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyCal.git")
  set(version "b78372e183")

  niftkMacroDefineExternalProjectVariables(NiftyCal ${version} ${location})
  set(proj_DEPENDENCIES OpenCV Eigen AprilTags ITK)

  if(NOT DEFINED NiftyCal_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${proj_LOCATION}
      GIT_TAG ${proj_VERSION}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_DEBUG_POSTFIX:STRING=
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DOpenCV_DIR:PATH=${OpenCV_DIR}
        -DEigen_DIR:PATH=${Eigen_DIR}
        -DEigen_INCLUDE_DIR:PATH=${Eigen_INCLUDE_DIR}
        -DAprilTags_DIRECTORY:PATH=${AprilTags_DIR}
        -DITK_DIR:PATH=${ITK_DIR}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NiftyCal_DIR ${proj_BUILD})
    set(NifTK_PREFIX_PATH ${NiftyCal_DIR}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading NiftyCal from ${NiftyCal_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif(BUILD_IGI)
