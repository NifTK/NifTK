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
find_package(AprilTags REQUIRED)
if(AprilTags_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${AprilTags_INCLUDE_DIR})
  list(APPEND ALL_LIBRARIES ${APRILTAGS_LIB})
  link_directories(${AprilTags_LIBRARY_DIR})
endif()
