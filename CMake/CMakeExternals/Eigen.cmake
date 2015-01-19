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

niftkMacroGetCommitHashOfCurrentFile(config_version)

set(proj Eigen)
set(proj_VERSION ${NIFTK_VERSION_EIGEN})
set(proj_SOURCE ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-src)
set(proj_CONFIG ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-cmake)
set(proj_BUILD ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-build)
set(proj_INSTALL ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-install)
set(proj_DEPENDENCIES )
set(Eigen_DEPENDS ${proj})

if(NOT DEFINED Eigen_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_EIGEN ${NIFTK_LOCATION_EIGEN})

  ExternalProject_Add(${proj}
    SOURCE_DIR ${proj_SOURCE}
    PREFIX ${proj_CONFIG}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${NIFTK_LOCATION_EIGEN}
    URL_MD5 ${NIFTK_CHECKSUM_EIGEN}
    #CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DEIGEN_LEAVE_TEST_IN_ALL_TARGET=ON
      -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(Eigen_DIR ${proj_SOURCE})
  set(Eigen_ROOT ${Eigen_DIR})
  set(Eigen_INCLUDE_DIR ${Eigen_DIR})

  message("SuperBuild loading Eigen from ${Eigen_DIR}")

else(NOT DEFINED Eigen_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED Eigen_DIR)
