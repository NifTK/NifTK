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

set(H_FILES
  niftkIBaseView.h
)

set(CPP_FILES
  Widgets/niftkDataStorageCheckableComboBox.cxx
  Widgets/niftkHelpAboutDialog.cxx
  Rendering/niftkSharedOGLContext.cxx
  Rendering/niftkScopedOGLContext.cxx
  niftkBaseController.cxx
  niftkBaseGUI.cxx
)

set(MOC_H_FILES
  Events/niftkPaintEventEater.h
  Events/niftkWheelEventEater.h
  Events/niftkMouseEventEater.h
  Widgets/niftkDataStorageCheckableComboBox.h
  Widgets/niftkHelpAboutDialog.h
  niftkBaseController.h
  niftkBaseGUI.h
)

set(UI_FILES
  Widgets/niftkHelpAboutDialog.ui
)

set(QRC_FILES
  Resources/niftkCoreGui.qrc
)

if(WIN32)
  set(CPP_FILES
    ${CPP_FILES}
    Events/niftkWindowsHotkeyHandler.cxx
  )
  set(MOC_H_FILES
    ${MOC_H_FILES}
    Events/niftkWindowsHotkeyHandler.h
  )
endif()
