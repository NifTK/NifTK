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
  INCLUDE_DIRS Interactions
  DEPENDS MitkCore MitkQtWidgets niftkITK niftkCore
  PACKAGE_DEPENDS Qt4|QtCore Qt5|Core CTK|CTKWidgets
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkDnDDisplay)
endif()

if(BUILD_DISPLAY_TESTING)
  add_subdirectory(Testing)
endif()
