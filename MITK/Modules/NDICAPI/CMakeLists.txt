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
  PACKAGE_DEPENDS Qt4|QtCore Qt5|Core
  INCLUDE_DIRS ndicapi
  NO_INIT      # Not an MITK MicroServices module
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkNDICAPI)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

