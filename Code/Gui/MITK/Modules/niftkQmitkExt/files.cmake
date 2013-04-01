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

SET(CPP_FILES
  LookupTables/LookupTableContainer.cxx
  LookupTables/LookupTableSaxHandler.cxx
  LookupTables/LookupTableManager.cxx
  Dialogs/QmitkHelpAboutDialog.cxx
  Dialogs/QmitkMIDASNewSegmentationDialog.cxx
  QmitkThumbnailRenderWindow.cxx
  QmitkMIDASToolSelectorWidget.cxx
  QmitkMIDASImageAndSegmentationSelectorWidget.cxx
  QmitkMIDASDrawToolGUI.cxx
  QmitkMIDASPaintbrushToolGUI.cxx
  QmitkMIDASBindWidget.cxx
  QmitkMIDASSlidersWidget.cxx
  QmitkMIDASOrientationWidget.cxx
  QmitkMIDASStdMultiWidget.cxx
  QmitkMIDASSingleViewWidget.cxx
  QmitkMIDASMultiViewVisibilityManager.cxx
  QmitkMIDASSingleViewWidgetListManager.cxx
  QmitkMIDASSingleViewWidgetListVisibilityManager.cxx
  QmitkMIDASSingleViewWidgetListDropManager.cxx
  QmitkSingleWidget.cxx
  QmitkCmicLogo.cxx
  QmitkBitmapOverlay.cxx
)

SET(MOC_H_FILES 
  Events/QmitkPaintEventEater.h
  Events/QmitkWheelEventEater.h
  Events/QmitkMouseEventEater.h
  Dialogs/QmitkHelpAboutDialog.h
  Dialogs/QmitkMIDASNewSegmentationDialog.h
  QmitkThumbnailRenderWindow.h
  QmitkMIDASToolSelectorWidget.h
  QmitkMIDASImageAndSegmentationSelectorWidget.h
  QmitkMIDASDrawToolGUI.h
  QmitkMIDASPaintbrushToolGUI.h
  QmitkMIDASBindWidget.h
  QmitkMIDASSlidersWidget.h
  QmitkMIDASOrientationWidget.h
  QmitkMIDASStdMultiWidget.h
  QmitkMIDASSingleViewWidget.h
  QmitkMIDASMultiViewVisibilityManager.h
  QmitkSingleWidget.h
)

SET(UI_FILES
  Resources/UI/QmitkHelpAboutDialog.ui
  Resources/UI/QmitkMIDASImageAndSegmentationSelector.ui
  Resources/UI/QmitkMIDASToolSelector.ui
  Resources/UI/QmitkMIDASBindWidget.ui
  Resources/UI/QmitkMIDASOrientationWidget.ui
  Resources/UI/QmitkMIDASSlidersWidget.ui
)

SET(QRC_FILES
  Resources/niftkQmitkExt.qrc
)
