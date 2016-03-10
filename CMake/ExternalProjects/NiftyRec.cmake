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
# NiftyRec
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NiftyRec_DIR AND NOT EXISTS ${NiftyRec_DIR})
  message(FATAL_ERROR "NiftyRec_DIR variable is defined but corresponds to non-existing directory \"${NiftyRec_DIR}\".")
endif()

if(BUILD_NiftyRec)

  set(version "14")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftyRec-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftyRec ${version} ${location})
  set(proj_DEPENDENCIES NiftyReg)

  if(NOT DEFINED NiftyRec_DIR)

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
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DCUDA_SDK_ROOT_DIR=${CUDA_SDK_ROOT_DIR}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NiftyRec_DIR ${proj_INSTALL})
#    set(NiftyRec_INCLUDE_DIR "${NiftyRec_DIR}/include")
#    set(NiftyRec_LIBRARY_DIR "${NiftyRec_DIR}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

    message("SuperBuild loading NiftyRec from ${NiftyRec_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif()
