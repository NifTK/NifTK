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
# NiftySeg
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYSEG_ROOT AND NOT EXISTS ${NIFTYSEG_ROOT})
  message(FATAL_ERROR "NIFTYSEG_ROOT variable is defined but corresponds to non-existing disegtory \"${NIFTYSEG_ROOT}\".")
endif()

if(BUILD_NIFTYSEG)

  set(version "b2decf5160")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftySeg-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftySeg ${version} ${location})
  set(proj_DEPENDENCIES Eigen)

  if(NOT DEFINED NIFTYSEG_ROOT)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${gen}
      #CONFIGURE_COMMAND ""
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      #BUILD_COMMAND ""
      #INSTALL_COMMAND ""
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DUSE_OPENMP:BOOL=OFF
        -DINSTALL_PRIORS:BOOL=ON
        -DINSTALL_PRIORS_DIRECTORY:PATH=${proj_INSTALL}/priors
        -DUSE_SYSTEM_EIGEN=ON
        -DEigen_INCLUDE_DIR=${Eigen_INCLUDE_DIR}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(NIFTYSEG_ROOT ${proj_INSTALL})
    set(NIFTYSEG_INCLUDE_DIR "${NIFTYSEG_ROOT}/include")
    set(NIFTYSEG_LIBRARY_DIR "${NIFTYSEG_ROOT}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading NiftySeg from ${NIFTYSEG_ROOT}")

  else(NOT DEFINED NIFTYSEG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYSEG_ROOT)

endif(BUILD_NIFTYSEG)
