SET(CPP_FILES
  Common/QmitkIGIUtils.cpp
  DataManagement/QmitkQImageToMitkImageFilter.cpp
  ToolsGui/TrackerControlsWidget.cpp
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.cpp
  ToolsGui/QmitkIGIDataSourceManager.cpp    
  ToolsGui/QmitkIGINiftyLinkDataType.cpp
  ToolsGui/QmitkIGINiftyLinkDataSource.cpp
  ToolsGui/QmitkIGIDataSourceGui.cpp
  ToolsGui/QmitkIGINiftyLinkDataSourceGui.cpp
  ToolsGui/QmitkIGITrackerTool.cpp
  ToolsGui/QmitkIGITrackerToolGui.cpp
  ToolsGui/QmitkIGIUltrasonixTool.cpp
  ToolsGui/QmitkIGIUltrasonixToolGui.cpp     
)

SET(MOC_H_FILES
  ToolsGui/TrackerControlsWidget.h
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.h
  ToolsGui/QmitkIGIDataSourceManager.h
  ToolsGui/QmitkIGINiftyLinkDataSource.h
  ToolsGui/QmitkIGIDataSourceGui.h
  ToolsGui/QmitkIGINiftyLinkDataSourceGui.h
  ToolsGui/QmitkIGITrackerTool.h
  ToolsGui/QmitkIGITrackerToolGui.h  
  ToolsGui/QmitkIGIUltrasonixTool.h
  ToolsGui/QmitkIGIUltrasonixToolGui.h
)

SET(UI_FILES
  ToolsGui/TrackerControlsWidget.ui
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.ui  
  ToolsGui/QmitkIGIToolManager.ui
  ToolsGui/QmitkIGITrackerToolGui.ui
  ToolsGui/QmitkIGIUltrasonixToolGui.ui
  ToolsGui/QmitkIGIDataSourceManager.ui
)

SET(QRC_FILES
#  Resources/niftkIGIGui.qrc
)