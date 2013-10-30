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

find_package(PCL 1.7 REQUIRED)
if(PCL_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${PCL_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${PCL_LIBRARIES})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif()

