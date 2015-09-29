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
if(DEFINED NiftySeg_DIR AND NOT EXISTS ${NiftySeg_DIR})
  message(FATAL_ERROR "NiftySeg_DIR variable is defined but corresponds to non-existing disegtory \"${NiftySeg_DIR}\".")
endif()

if(BUILD_NiftySeg)

  set(version "d7ba4fa396")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NiftySeg-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(NiftySeg ${version} ${location})
  set(proj_DEPENDENCIES Eigen)

  if(NOT DEFINED NiftySeg_DIR)

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

    set(NiftySeg_DIR ${proj_INSTALL})
#    set(NiftySeg_INCLUDE_DIR "${NiftySeg_DIR}/include")
#    set(NiftySeg_LIBRARY_DIR "${NiftySeg_DIR}/lib")

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading NiftySeg from ${NiftySeg_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()

endif()
