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
find_package(Eigen REQUIRED)
if(Eigen_FOUND)
  message("Found Eigen in ${Eigen_DIR}")
  list(APPEND ALL_INCLUDE_DIRECTORIES ${Eigen_INCLUDE_DIR})
endif()
