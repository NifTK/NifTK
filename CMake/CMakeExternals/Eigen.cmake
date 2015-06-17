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
# Eigen
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED Eigen_DIR AND NOT EXISTS ${Eigen_DIR})
  message(FATAL_ERROR "Eigen_DIR variable is defined but corresponds to non-existing directory \"${Eigen_ROOT}\".")
endif()

set(version "3.2.2.1")
set(location "${NIFTK_EP_TARBALL_LOCATION}/eigen-eigen-${version}.tar.bz2")

niftkMacroDefineExternalProjectVariables(Eigen ${version} ${location})

if(NOT DEFINED Eigen_DIR)

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
      -DEIGEN_LEAVE_TEST_IN_ALL_TARGET=ON
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(Eigen_DIR ${proj_SOURCE})
  set(Eigen_ROOT ${Eigen_DIR})
  set(Eigen_INCLUDE_DIR ${Eigen_DIR})

  set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

  message("SuperBuild loading Eigen from ${Eigen_DIR}")

else(NOT DEFINED Eigen_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED Eigen_DIR)
