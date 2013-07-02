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
if(DEFINED aruco_DIR AND NOT EXISTS ${aruco_DIR})
  message(FATAL_ERROR "aruco_DIR variable is defined but corresponds to non-existing directory \"${aruco_DIR}\".")
endif()

set(proj aruco)
set(proj_DEPENDENCIES OpenCV)
set(aruco_DEPENDS ${proj})

if(NOT DEFINED aruco_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_ARUCO ${NIFTK_LOCATION_ARUCO})

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_ARUCO}
    URL_MD5 ${NIFTK_CHECKSUM_ARUCO}
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
        -DOpenCV_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/OpenCV-build
     DEPENDS ${proj_DEPENDENCIES}
    )

  set(aruco_DIR ${EP_BASE}/Install/${proj})
  message("SuperBuild loading ARUCO from ${aruco_DIR}")

else(NOT DEFINED aruco_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED aruco_DIR)
