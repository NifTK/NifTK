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

  if(WIN32)
    # Force dynamic linking
    list(APPEND ALL_COMPILE_OPTIONS -DBOOST_ALL_DYN_LINK)
  else()
    # Boost has an auto link feature (pragma comment lib) for Windows
    list(APPEND ALL_LIBRARIES ${Boost_LIBRARIES})
  endif()
endif()
