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
# aruco - external project for tracking AR markers.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED aruco_DIR AND NOT EXISTS ${aruco_DIR})
  message(FATAL_ERROR "aruco_DIR variable is defined but corresponds to non-existing directory \"${aruco_DIR}\".")
endif()

if(BUILD_IGI)

  set(version "1.2.4")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/aruco-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(ArUco ${version} ${location})
  set(proj_DEPENDENCIES OpenCV)

  if(NOT DEFINED aruco_DIR)

    niftkMacroGetChecksum(proj_CHECKSUM ${proj_LOCATION})

    ExternalProject_Add(${proj}
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DOpenCV_DIR:PATH=${OpenCV_DIR}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(aruco_DIR ${proj_INSTALL})
    message("SuperBuild loading ArUco from ${aruco_DIR}")

  else(NOT DEFINED aruco_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED aruco_DIR)

endif()


