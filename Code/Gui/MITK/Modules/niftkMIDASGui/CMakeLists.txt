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

MITK_CREATE_MODULE(niftkMIDASGui
  EXPORT_DEFINE NIFTKMIDASGUI_EXPORT
  INCLUDE_DIRS Events Dialogs LookupTables
  DEPENDS Mitk Qmitk niftkCore niftkCoreGui niftkMIDAS
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

TARGET_LINK_LIBRARIES(niftkMIDASGui
  niftkITK
  )
