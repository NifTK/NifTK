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
  DataManagement/QmitkQImageToMitkImageFilter.cxx
  ToolsGui/TrackerControlsWidget.cxx
  ToolsGui/QmitkIGITimerBasedThread.cxx
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.cxx
  ToolsGui/QmitkIGINiftyLinkDataType.cxx
  ToolsGui/QmitkIGINiftyLinkDataSource.cxx
  ToolsGui/QmitkIGINiftyLinkDataSourceGui.cxx
  ToolsGui/QmitkIGIDataSource.cxx
  ToolsGui/QmitkIGIDataSourceBackgroundSaveThread.cxx
  ToolsGui/QmitkIGIDataSourceGui.cxx
  ToolsGui/QmitkIGILocalDataSource.cxx
  ToolsGui/QmitkIGILocalDataSourceGrabbingThread.cxx
  ToolsGui/QmitkIGITrackerTool.cxx
  ToolsGui/QmitkIGITrackerToolGui.cxx
  ToolsGui/QmitkIGIUltrasonixTool.cxx
  ToolsGui/QmitkIGIUltrasonixToolGui.cxx
  ToolsGui/QmitkIGIOpenCVDataSource.cxx
  ToolsGui/QmitkIGIOpenCVDataSourceGui.cxx
)

SET(MOC_H_FILES
  ToolsGui/TrackerControlsWidget.h
  ToolsGui/QmitkIGITimerBasedThread.h
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.h
  ToolsGui/QmitkIGINiftyLinkDataSource.h
  ToolsGui/QmitkIGINiftyLinkDataSourceGui.h
  ToolsGui/QmitkIGIDataSource.h
  ToolsGui/QmitkIGIDataSourceGui.h
  ToolsGui/QmitkIGILocalDataSource.h
  ToolsGui/QmitkIGITrackerTool.h
  ToolsGui/QmitkIGITrackerToolGui.h  
  ToolsGui/QmitkIGIUltrasonixTool.h
  ToolsGui/QmitkIGIUltrasonixToolGui.h
  ToolsGui/QmitkIGIOpenCVDataSource.h
  ToolsGui/QmitkIGIOpenCVDataSourceGui.h
)

SET(UI_FILES
  ToolsGui/TrackerControlsWidget.ui
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.ui  
  ToolsGui/QmitkIGITrackerToolGui.ui
  ToolsGui/QmitkIGIUltrasonixToolGui.ui
)

SET(QRC_FILES
  #Resources/niftkIGIGui.qrc
)
