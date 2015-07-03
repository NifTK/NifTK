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
if(DEFINED NIFTYREC_ROOT AND NOT EXISTS ${NIFTYREC_ROOT})
  message(FATAL_ERROR "NIFTYREC_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREC_ROOT}\".")
endif()

if(BUILD_NIFTYREC)

  set(version "14")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftyRec-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftyRec ${version} ${location})
  set(proj_DEPENDENCIES NiftyReg)

  if(NOT DEFINED NIFTYREC_ROOT)

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

    set(NIFTYREC_ROOT ${proj_INSTALL})
    set(NIFTYREC_INCLUDE_DIR "${NIFTYREC_ROOT}/include")
    set(NIFTYREC_LIBRARY_DIR "${NIFTYREC_ROOT}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

    message("SuperBuild loading NiftyRec from ${NIFTYREC_ROOT}")

  else(NOT DEFINED NIFTYREC_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYREC_ROOT)

endif(BUILD_NIFTYREC)
