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

# Add more COMPONENTS as and when we need them.
find_package(PCL REQUIRED COMPONENTS common)

if(PCL_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${PCL_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${PCL_LIBRARIES})
  link_directories(${PCL_LIBRARY_DIRS})
endif()

