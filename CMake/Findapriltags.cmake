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

  
set(apriltags_FOUND 0)

set(apriltags_DIR @apriltags_DIR@)

find_path(APRILTAGS_INC
  Tag16h5.h 
  PATHS ${apriltags_DIR}/include/AprilTags
  NO_DEFAULT_PATH
  )

find_library(APRILTAGS_LIB
  apriltags
  PATHS ${apriltags_DIR}/lib
  NO_DEFAULT_PATH
  )

if(APRILTAGS_INC AND APRILTAGS_LIB)
  set(apriltags_FOUND 1)

  get_filename_component(_inc_dir ${APRILTAGS_INC} PATH)
  set(apriltags_INCLUDE_DIR ${_inc_dir})

  set(apriltags_LIBRARIES ${APRILTAGS_LIB})

  get_filename_component(_lib_dir ${APRILTAGS_LIB} PATH)
  set(apriltags_LIBRARY_DIR ${_lib_dir})

endif()
