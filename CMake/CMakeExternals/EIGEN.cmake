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
# EIGEN
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED EIGEN_DIR AND NOT EXISTS ${EIGEN_DIR})
  message(FATAL_ERROR "EIGEN_DIR variable is defined but corresponds to non-existing directory \"${EIGEN_ROOT}\".")
endif()

if(BUILD_IGI)

  set(proj EIGEN)
  set(proj_DEPENDENCIES )
  set(EIGEN_DEPENDS ${proj})

  if(NOT DEFINED EIGEN_DIR)

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

    set(EIGEN_DIR ${CMAKE_BINARY_DIR}/${proj}-src)
    set(EIGEN_ROOT ${EIGEN_DIR})

    message("SuperBuild loading EIGEN from ${EIGEN_DIR}")

  else(NOT DEFINED EIGEN_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED EIGEN_DIR)

endif(BUILD_IGI)
