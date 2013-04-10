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

MITK_CREATE_MODULE(niftkIGIGui
  EXPORT_DEFINE NIFTKIGIGUI_EXPORT
  INCLUDE_DIRS ToolsGui Common DataManagement
  DEPENDS MitkExt QmitkExt MitkIGTUI niftkCore niftkIGI
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

IF(BUILD_TESTING)  
  ADD_SUBDIRECTORY(Testing)
ENDIF()
