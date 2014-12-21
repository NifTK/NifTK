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


set(Eigen_FOUND 0)
set(Eigen_DIR @Eigen_DIR@)
if(Eigen_DIR AND EXISTS "@Eigen_DIR@")
  set(Eigen_FOUND 1)
endif()
