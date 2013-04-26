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
  DataSources/QmitkIGIUltrasonixTool.cxx
  DataSources/QmitkIGIUltrasonixToolGui.cxx
  DataSources/QmitkIGIOpenCVDataSource.cxx
  DataSources/QmitkIGIOpenCVDataSourceGui.cxx
)

SET(MOC_H_FILES
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
  DataSources/QmitkIGIUltrasonixTool.h
  DataSources/QmitkIGIUltrasonixToolGui.h
  DataSources/QmitkIGIOpenCVDataSource.h
  DataSources/QmitkIGIOpenCVDataSourceGui.h
)

SET(UI_FILES
  DataSources/TrackerControlsWidget.ui
  DataSources/QmitkFiducialRegistrationWidgetDialog.ui  
  DataSources/QmitkIGITrackerToolGui.ui
  DataSources/QmitkIGIUltrasonixToolGui.ui
)

SET(QRC_FILES
  #Resources/niftkIGIGui.qrc
)
