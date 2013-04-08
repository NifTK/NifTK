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
# ARUCO - external project for tracking AR markers.
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED ARUCO_DIR AND NOT EXISTS ${ARUCO_DIR})
  MESSAGE(FATAL_ERROR "ARUCO_DIR variable is defined but corresponds to non-existing directory \"${ARUCO_DIR}\".")
ENDIF()

SET(proj ARUCO)
SET(proj_DEPENDENCIES MITK)
SET(ARUCO_DEPENDS ${proj})

IF(NOT DEFINED ARUCO_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_ARUCO ${NIFTK_LOCATION_ARUCO})

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_ARUCO}
    URL_MD5 ${NIFTK_CHECKSUM_ARUCO}
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
        -DOpenCV_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/MITK-build/OpenCV-build
     DEPENDS ${proj_DEPENDENCIES}
    )

  SET(ARUCO_DIR ${EP_BASE}/Install/${proj})
  MESSAGE("SuperBuild loading ARUCO from ${ARUCO_DIR}")

ELSE(NOT DEFINED ARUCO_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED ARUCO_DIR)
