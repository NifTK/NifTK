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
  DEPENDS Mitk MitkExt Qmitk QmitkExt MitkIGT MitkIGTUI niftkMitkExt niftkIGI
  PACKAGE_DEPENDS QT QVTK CTK NiftyLink
  QT_MODULE 
)

ADD_SUBDIRECTORY(Testing)
