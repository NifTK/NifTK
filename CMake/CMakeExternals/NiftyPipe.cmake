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
# NiftyPipe
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NiftyPipe_DIR AND NOT EXISTS ${NiftyPipe_DIR})
  message(FATAL_ERROR "NiftyPipe_DIR variable is defined but corresponds to non-existing directory \"${NiftyPipe_DIR}\".")
endif()

if(BUILD_NiftyPipe)
  
  set(version "6566c3d918")
  set(location "git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe")

  niftkMacroDefineExternalProjectVariables(NiftyPipe ${version} ${location})
  set(proj_DEPENDENCIES )

  if(NOT DEFINED NiftyPipe_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${proj_LOCATION}
      GIT_TAG ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
      )
    set(NiftyPipe_DIR ${proj_INSTALL})
    set(NiftyPipe_ROOT ${proj_INSTALL})

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading NiftyPipe from ${NiftyPipe_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif()
