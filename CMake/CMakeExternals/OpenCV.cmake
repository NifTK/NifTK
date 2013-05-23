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
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_DOXYGEN_DOCS:BOOL=OFF
    -DWITH_CUDA:BOOL=OFF
    -DWITH_QT:BOOL=OFF
    -DADDITIONAL_C_FLAGS:STRING=${NIFTK_ADDITIONAL_C_FLAGS}
    -DADDITIONAL_CXX_FLAGS:STRING=${NIFTK_ADDITIONAL_CXX_FLAGS}
    DEPENDS ${proj_DEPENDENCIES}
  )
  set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  message("SuperBuild loading OpenCV from ${OpenCV_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
