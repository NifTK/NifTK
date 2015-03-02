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
# NiftySim
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYSIM_ROOT AND NOT EXISTS ${NIFTYSIM_ROOT})
  message(FATAL_ERROR "NIFTYSIM_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYSIM_ROOT}\".")
endif()

if(BUILD_NIFTYSIM)

  set(version "0d6fa842e6")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftySim-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftySim ${version} ${location})

  if(NOT DEFINED NIFTYSIM_ROOT)
    if(DEFINED VTK_DIR)
      set(USE_VTK ON)
    else(DEFINED VTK_DIR)
      set(USE_VTK OFF)
    endif(DEFINED VTK_DIR)

    if (USE_VTK)
      list(APPEND proj_DEPENDENCIES VTK)
    endif (USE_VTK)

    # Run search for needed CUDA SDK components here so as to give the user the option
    # to manually set paths should they not be found by NiftySim.
    if (NIFTK_USE_CUDA)
      if (CUDA_VERSION VERSION_GREATER "5.0" OR CUDA_VERSION VERSION_EQUAL "5.0")
        find_path(CUDA_SDK_COMMON_INCLUDE_DIR
          helper_cuda.h
          PATHS ${CUDA_SDK_SEARCH_PATH}
          PATH_SUFFIXES "common/inc"
          DOC "Location of helper_cuda.h"
          NO_DEFAULT_PATH
        )
        mark_as_advanced(CUDA_SDK_COMMON_INCLUDE_DIR)
      else ()
        find_path(CUDA_CUT_INCLUDE_DIR
          cutil.h
          PATHS ${CUDA_SDK_SEARCH_PATH}
          PATH_SUFFIXES "common/inc"
          DOC "Location of cutil.h"
          NO_DEFAULT_PATH
        )
        mark_as_advanced(CUDA_CUT_INCLUDE_DIR)
      endif (CUDA_VERSION VERSION_GREATER "5.0" OR CUDA_VERSION VERSION_EQUAL "5.0")
    else ()
      if (NIFTYSIM_USE_CUDA)
        message(FATAL_ERROR "In order to use CUDA in NiftySim you must enable CUDA support in NifTK.")
      endif (NIFTYSIM_USE_CUDA)
    endif (NIFTK_USE_CUDA)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=${NIFTYSIM_USE_CUDA}
        -DCUDA_CUT_INCLUDE_DIR:STRING=${CUDA_CUT_INCLUDE_DIR}
        -DUSE_BOOST:BOOL=OFF
        -DUSE_VIZ:BOOL=${USE_VTK}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DCUDA_TOOLKIT_ROOT_DIR:PATH=${CUDA_TOOLKIT_ROOT_DIR}
        -DCUDA_SDK_ROOT_DIR:PATH=${CUDA_SDK_ROOT_DIR}
        -DCUDA_CUT_INCLUDE_DIR:PATH=${CUDA_CUT_INCLUDE_DIR}
        -DCUDA_COMMON_INCLUDE_DIR:PATH=${CUDA_COMMON_INCLUDE_DIR}
        -DCUDA_HOST_COMPILER:PATH=${CUDA_HOST_COMPILER}
        -DCMAKE_CXX_FLAGS:STRING=${NIFTYSIM_CMAKE_CXX_FLAGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NIFTYSIM_ROOT ${proj_INSTALL})
    set(NIFTYSIM_INCLUDE_DIR "${NIFTYSIM_ROOT}/include")
    set(NIFTYSIM_LIBRARY_DIR "${NIFTYSIM_ROOT}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

    message("SuperBuild loading NiftySim from ${NIFTYSIM_ROOT}")

  else(NOT DEFINED NIFTYSIM_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYSIM_ROOT)

endif(BUILD_NIFTYSIM)
