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
# NiftyReg
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NiftyReg_DIR AND NOT EXISTS ${NiftyReg_DIR})
  message(FATAL_ERROR "NiftyReg_DIR variable is defined but corresponds to non-existing directory \"${NiftyReg_DIR}\".")
endif()

if(BUILD_NiftyReg)

  set(version "97383b06b9")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftyReg-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftyReg ${version} ${location})

  if(NOT DEFINED NiftyReg_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBUILD_ALL_DEP:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=OFF
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NiftyReg_DIR ${proj_INSTALL})

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading NiftyReg from ${NiftyReg_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif(BUILD_NiftyReg)
