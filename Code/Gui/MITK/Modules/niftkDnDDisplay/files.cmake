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
  QmitkMIDASSlidersWidget.cxx
  QmitkMIDASLayoutWidget.cxx
  niftkMultiWindowWidget.cxx
  niftkMultiViewerVisibilityManager.cxx
  niftkMultiViewerWidget.cxx
  niftkMultiViewerWidgetControlPanel.cxx
  niftkSingleViewerWidget.cxx
  niftkSingleViewerWidgetListManager.cxx
  niftkSingleViewerWidgetListVisibilityManager.cxx
  niftkSingleViewerWidgetListDropManager.cxx
  vtkSideAnnotation.cxx
  Interactions/mitkMIDASViewKeyPressStateMachine.cxx
  Interactions/mitkMIDASDisplayInteractor.cxx
)

set(MOC_H_FILES 
  QmitkMIDASSlidersWidget.h
  QmitkMIDASLayoutWidget.h
  niftkMultiViewerWidget.h
  niftkMultiViewerWidgetControlPanel.h
  niftkMultiWindowWidget.h
  niftkSingleViewerWidget.h
  niftkMultiViewerVisibilityManager.h
)

set(UI_FILES
  Resources/UI/QmitkMIDASLayoutWidget.ui
  Resources/UI/QmitkMIDASSlidersWidget.ui
  Resources/UI/niftkMultiViewerWidgetControlPanel.ui
)

set(RESOURCE_FILES
  Interactions/DisplayInteraction.xml
)

set(QRC_FILES
  Resources/niftkDnDDisplay.qrc
)
