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

MITK_CREATE_MODULE(niftkCoreGui
  EXPORT_DEFINE NIFTKCOREGUI_EXPORT
  INCLUDE_DIRS Events Dialogs LookupTables
  DEPENDS Mitk MitkExt Qmitk QmitkExt niftkCore
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

TARGET_LINK_LIBRARIES(niftkCoreGui
  niftkITK
  
  )

IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)
