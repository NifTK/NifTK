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

if(Boost_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${Boost_LIBRARIES})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

