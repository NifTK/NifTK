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


MITK_CREATE_MODULE(niftkPointReg
  INCLUDE_DIRS Registration Maths Utilities
  DEPENDS niftkCore niftkOpenCVUtils MitkIGT
  PACKAGE_DEPENDS OpenCV
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkPointReg)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
