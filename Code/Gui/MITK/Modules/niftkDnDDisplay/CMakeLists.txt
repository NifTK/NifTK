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
  DEPENDS MitkCore MitkQtWidgets niftkCore
  PACKAGE_DEPENDS Qt4|QtCore CTK|CTKWidgets
)

target_link_libraries(niftkDnDDisplay
  niftkITK
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkDnDDisplay)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
