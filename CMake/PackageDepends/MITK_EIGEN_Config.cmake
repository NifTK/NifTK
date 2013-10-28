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
find_package(EIGEN REQUIRED)
if(EIGEN_FOUND)
  message("Found EIGEN in ${EIGEN_DIR}")
  list(APPEND ALL_INCLUDE_DIRECTORIES ${EIGEN_INCLUDE_DIR})
endif()
