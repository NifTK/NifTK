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

MITK_CREATE_MODULE(
  INCLUDE_DIRS TrackedImage TrackedPointer Utils Rendering
  DEPENDS niftkCore niftkOpenCV niftkOpenCVUtils
  PACKAGE_DEPENDS Qt4|QtCore Qt5|Core
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGI)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
