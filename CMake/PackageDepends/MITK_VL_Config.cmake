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

# only try to find the VL package if not done already on the top-level cmakelists.
if(NOT VL_FOUND)
  find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLVivid VLQt4 REQUIRED)
endif()

if(VL_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${VL_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${VL_LIBRARIES})
  include_directories(${VL_INCLUDE_DIRS})
  link_directories(${VL_LIBRARY_DIRS})
endif()
