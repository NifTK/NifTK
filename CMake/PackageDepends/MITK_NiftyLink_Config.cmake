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

if(NiftyLink_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${NiftyLink_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${NiftyLink_LIBRARIES})
  link_directories(${NiftyLink_LIBRARY_DIRS})
endif()

