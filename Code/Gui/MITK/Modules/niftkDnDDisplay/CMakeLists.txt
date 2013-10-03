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

MITK_CREATE_MODULE(niftkDnDDisplay
  EXPORT_DEFINE NIFTKDNDDISPLAY_EXPORT
  INCLUDE_DIRS Events Dialogs LookupTables
  DEPENDS Mitk Qmitk niftkCore niftkCoreGui
  PACKAGE_DEPENDS QT QVTK CTK
  QT_MODULE 
)

target_link_libraries(niftkDnDDisplay
  niftkITK
  )
  
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
