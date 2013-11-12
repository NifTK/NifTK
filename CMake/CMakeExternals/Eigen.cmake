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

if(BUILD_IGI)

  set(proj Eigen)
  set(proj_DEPENDENCIES )
  set(Eigen_DEPENDS ${proj})

  if(NOT DEFINED Eigen_DIR)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_EIGEN ${NIFTK_LOCATION_EIGEN})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_EIGEN}
      URL_MD5 ${NIFTK_CHECKSUM_EIGEN}
      CONFIGURE_COMMAND ""
      UPDATE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      DEPENDS ${proj_DEPENDENCIES}
      )

    set(Eigen_DIR ${CMAKE_BINARY_DIR}/${proj}-src)
    set(Eigen_ROOT ${Eigen_DIR})

    message("SuperBuild loading Eigen from ${Eigen_DIR}")

  else(NOT DEFINED Eigen_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED Eigen_DIR)

endif(BUILD_IGI)
