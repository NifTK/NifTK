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

set(CPP_FILES
  LookupTables/LookupTableContainer.cxx
  LookupTables/LookupTableSaxHandler.cxx
  LookupTables/LookupTableManager.cxx
  Dialogs/QmitkHelpAboutDialog.cxx
  QmitkSingleWidget.cxx
  QmitkCmicLogo.cxx
  QmitkBitmapOverlay.cxx
)

set(MOC_H_FILES
  Events/QmitkPaintEventEater.h
  Events/QmitkWheelEventEater.h
  Events/QmitkMouseEventEater.h
  Dialogs/QmitkHelpAboutDialog.h
  QmitkSingleWidget.h
)

set(UI_FILES
  Resources/UI/QmitkHelpAboutDialog.ui
)

set(QRC_FILES
  Resources/niftkCoreGui.qrc
)
