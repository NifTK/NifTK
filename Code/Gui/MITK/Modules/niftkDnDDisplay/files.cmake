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
  niftkImageNavigatorWidget.cxx
  niftkWindowLayoutWidget.cxx
  niftkMultiWindowWidget.cxx
  niftkMultiViewerVisibilityManager.cxx
  niftkMultiViewerWidget.cxx
  niftkMultiViewerWidgetControlPanel.cxx
  niftkSingleViewerWidget.cxx
  vtkSideAnnotation.cxx
  Interactions/mitkDnDDisplayStateMachine.cxx
  Interactions/mitkDnDDisplayInteractor.cxx
)

set(MOC_H_FILES 
  niftkImageNavigatorWidget.h
  niftkWindowLayoutWidget.h
  niftkMultiViewerWidget.h
  niftkMultiViewerWidgetControlPanel.h
  niftkMultiWindowWidget.h
  niftkSingleViewerWidget.h
  niftkMultiViewerVisibilityManager.h
)

set(UI_FILES
  Resources/UI/niftkWindowLayoutWidget.ui
  Resources/UI/niftkImageNavigatorWidget.ui
  Resources/UI/niftkMultiViewerWidgetControlPanel.ui
)

set(RESOURCE_FILES
  Interactions/DnDDisplayInteraction.xml
)

set(QRC_FILES
  Resources/niftkDnDDisplay.qrc
)
