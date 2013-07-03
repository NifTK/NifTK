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
# OpenCV
#-----------------------------------------------------------------------------


# Sanity checks
if(DEFINED OpenCV_DIR AND NOT EXISTS ${OpenCV_DIR})
  message(FATAL_ERROR "OpenCV_DIR variable is defined but corresponds to non-existing directory")
endif()

set(proj OpenCV)
set(proj_DEPENDENCIES)
set(OpenCV_DEPENDS ${proj})

if(NOT DEFINED OpenCV_DIR)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_OPENCV ${NIFTK_LOCATION_OPENCV})
  
  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_OPENCV}
    URL_MD5 ${NIFTK_CHECKSUM_OPENCV}
    BINARY_DIR ${proj}-build
    UPDATE_COMMAND  ""
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_CACHE_ARGS
    ${EP_COMMON_ARGS}
    -DBUILD_opencv_core:BOOL=ON
    -DBUILD_opencv_calib3d:BOOL=ON
    -DBUILD_opencv_features2d:BOOL=ON
    -DBUILD_opencv_imgproc:BOOL=ON
    -DBUILD_opencv_video:BOOL=ON
    -DBUILD_opencv_python:BOOL=OFF
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_DOXYGEN_DOCS:BOOL=OFF
    -DBUILD_PERF_TESTS:BOOL=OFF
    -DWITH_CUDA:BOOL=${OPENCV_WITH_CUDA}
    -DWITH_QT:BOOL=OFF
    -DWITH_FFMPEG:BOOL=OFF
    -DADDITIONAL_C_FLAGS:STRING=${OPENCV_ADDITIONAL_C_FLAGS}
    -DADDITIONAL_CXX_FLAGS:STRING=${OPENCV_ADDITIONAL_CXX_FLAGS}
    DEPENDS ${proj_DEPENDENCIES}
  )
  set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  message("SuperBuild loading OpenCV from ${OpenCV_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
