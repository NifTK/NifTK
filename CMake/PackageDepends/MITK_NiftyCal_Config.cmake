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
find_package(NiftyCal REQUIRED)
if(NiftyCal_FOUND)
  include_directories(${NiftyCal_INCLUDE_DIRS})
  list(APPEND ALL_INCLUDE_DIRECTORIES ${NiftyCal_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${NiftyCal_LIBRARIES})
endif()

