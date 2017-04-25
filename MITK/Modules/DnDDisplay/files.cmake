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
  niftkDnDDisplayEnums.h
)

set(CPP_FILES
  niftkClickableLabel.cxx
  niftkMultiWindowWidget.cxx
  niftkMultiViewerVisibilityManager.cxx
  niftkMultiViewerWidget.cxx
  niftkMultiViewerControls.cxx
  niftkSingleViewerControls.cxx
  niftkSingleViewerWidget.cxx
  Interactions/niftkDnDDisplayInteractor.cxx
  Interactions/niftkVtkInteractorStyle.cxx
)

set(MOC_H_FILES 
  niftkClickableLabel_p.h
  niftkMultiViewerWidget.h
  niftkMultiViewerControls.h
  niftkSingleViewerControls.h
  niftkMultiWindowWidget_p.h
  niftkSingleViewerWidget.h
  niftkMultiViewerVisibilityManager.h
  Interactions/niftkDnDDisplayInteractor.h
)

set(UI_FILES
  Resources/UI/niftkMultiViewerControls.ui
  Resources/UI/niftkSingleViewerControls.ui
)

set(RESOURCE_FILES
  Interactions/DnDDisplayConfig.xml
  Interactions/DnDDisplayInteraction.xml
)

set(QRC_FILES
  Resources/niftkDnDDisplay.qrc
)
