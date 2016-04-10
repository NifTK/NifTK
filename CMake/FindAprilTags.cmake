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


set(AprilTags_FOUND 0)

set(AprilTags_DIR @AprilTags_DIRECTORY@)

find_path(APRILTAGS_INC
  Tag16h5.h 
  PATHS ${AprilTags_DIR}/include/AprilTags
  NO_DEFAULT_PATH
  )

find_library(APRILTAGS_LIB
  apriltags
  PATHS ${AprilTags_DIR}/lib
  NO_DEFAULT_PATH
  )

if(APRILTAGS_INC AND APRILTAGS_LIB)
  set(AprilTags_FOUND 1)

  get_filename_component(_inc_dir ${APRILTAGS_INC} PATH)
  set(AprilTags_INCLUDE_DIR ${_inc_dir})

  set(AprilTags_LIBRARIES ${APRILTAGS_LIB})

  get_filename_component(_lib_dir ${APRILTAGS_LIB} PATH)
  set(AprilTags_LIBRARY_DIR ${_lib_dir})

endif()
