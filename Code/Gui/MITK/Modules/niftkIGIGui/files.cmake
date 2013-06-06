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
  Common/QmitkIGIUtils.cxx
  DataSources/QmitkQImageToMitkImageFilter.cxx
  DataSources/TrackerControlsWidget.cxx
  DataSources/QmitkIGITimerBasedThread.cxx
  DataSources/QmitkFiducialRegistrationWidgetDialog.cxx
  DataSources/QmitkIGINiftyLinkDataType.cxx
  DataSources/QmitkIGINiftyLinkDataSource.cxx
  DataSources/QmitkIGINiftyLinkDataSourceGui.cxx
  DataSources/QmitkIGIDataSource.cxx
  DataSources/QmitkIGIDataSourceBackgroundSaveThread.cxx
  DataSources/QmitkIGIDataSourceGui.cxx
  DataSources/QmitkIGILocalDataSource.cxx
  DataSources/QmitkIGILocalDataSourceGrabbingThread.cxx
  DataSources/QmitkIGITrackerTool.cxx
  DataSources/QmitkIGITrackerToolGui.cxx
  DataSources/QmitkIGITrackerSource.cxx
  DataSources/QmitkIGITrackerSourceGui.cxx
  DataSources/QmitkIGIUltrasonixTool.cxx
  DataSources/QmitkIGIUltrasonixToolGui.cxx
  DataSources/QmitkIGIOpenCVDataSource.cxx
  DataSources/QmitkIGIOpenCVDataSourceGui.cxx
  OverlayEditor/QmitkBitmapOverlay.cxx
  OverlayEditor/QmitkSingle3DView.cxx
  OverlayEditor/QmitkIGIOverlayEditor.cxx
)

set(MOC_H_FILES
  DataSources/TrackerControlsWidget.h
  DataSources/QmitkIGITimerBasedThread.h
  DataSources/QmitkFiducialRegistrationWidgetDialog.h
  DataSources/QmitkIGINiftyLinkDataSource.h
  DataSources/QmitkIGINiftyLinkDataSourceGui.h
  DataSources/QmitkIGIDataSource.h
  DataSources/QmitkIGIDataSourceGui.h
  DataSources/QmitkIGILocalDataSource.h
  DataSources/QmitkIGITrackerTool.h
  DataSources/QmitkIGITrackerToolGui.h  
  DataSources/QmitkIGITrackerSource.h
  DataSources/QmitkIGITrackerSourceGui.h
  DataSources/QmitkIGIUltrasonixTool.h
  DataSources/QmitkIGIUltrasonixToolGui.h
  DataSources/QmitkIGIOpenCVDataSource.h
  DataSources/QmitkIGIOpenCVDataSourceGui.h
  OverlayEditor/QmitkSingle3DView.h
  OverlayEditor/QmitkIGIOverlayEditor.h
)

set(UI_FILES
  DataSources/TrackerControlsWidget.ui
  DataSources/QmitkFiducialRegistrationWidgetDialog.ui  
  DataSources/QmitkIGITrackerToolGui.ui
  DataSources/QmitkIGITrackerSourceGui.ui
  DataSources/QmitkIGIUltrasonixToolGui.ui
  OverlayEditor/QmitkIGIOverlayEditor.ui
)

set(QRC_FILES
  #Resources/niftkIGIGui.qrc
)
