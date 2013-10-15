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


set(EIGEN_FOUND 0)
set(EIGEN_DIR @EIGEN_DIR@)
if(EIGEN_DIR)
  set(EIGEN_INCLUDE_DIR ${EIGEN_DIR})
  set(EIGEN_FOUND 1)
endif()
