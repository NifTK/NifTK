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
  INCLUDE_DIRS Events Rendering Widgets
  DEPENDS MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore
  PACKAGE_DEPENDS Qt4|QtGui Qt4|QtOpenGL Qt5|Gui Qt5|OpenGL CTK|CTKWidgets
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreGui)
endif()

if(BUILD_TESTING)
  #add_subdirectory(Testing)
endif(BUILD_TESTING)
