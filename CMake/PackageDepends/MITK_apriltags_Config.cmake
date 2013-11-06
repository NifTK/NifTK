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
find_package(apriltags REQUIRED)
if(apriltags_FOUND)
  message("Found AprilTags include dir=${apriltags_INCLUDE_DIR}")
  message("Found AprilTags lib dir    =${apriltags_LIBRARY_DIR}")
  message("Found AprilTags library    =${apriltags_LIBRARIES}")
  list(APPEND ALL_INCLUDE_DIRECTORIES ${apriltags_INCLUDE_DIR})
  list(APPEND ALL_LIBRARIES ${APRILTAGS_LIB})
  link_directories(${apriltags_LIBRARY_DIR})
endif()
