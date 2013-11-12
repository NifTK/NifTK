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
find_package(FLANN REQUIRED)
if(FLANN_FOUND)
  list(APPEND ALL_INCLUDE_DIRECTORIES ${FLANN_DIR}/include)
  list(APPEND ALL_LIBRARIES flann_cpp)
  link_directories(${FLANN_DIR}/lib)
endif()
