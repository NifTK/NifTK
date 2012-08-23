SET(CPP_FILES
  Common/QmitkIGIUtils.cpp
  ToolsGui/TrackerControlsWidget.cpp
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.cpp
  ToolsGui/QmitkIGITool.cpp
  ToolsGui/QmitkIGIToolGui.cpp
  ToolsGui/QmitkIGIToolFactory.cpp  
  ToolsGui/QmitkIGIToolManager.cpp  
  ToolsGui/QmitkIGITrackerTool.cpp
  ToolsGui/QmitkIGITrackerToolGui.cpp
  ToolsGui/QmitkIGIUltrasonixTool.cpp
  ToolsGui/QmitkIGIUltrasonixToolGui.cpp   
)

SET(MOC_H_FILES
  ToolsGui/TrackerControlsWidget.h
  ToolsGui/QmitkFiducialRegistrationWidgetDialog.h
  ToolsGui/QmitkIGITool.h  
  ToolsGui/QmitkIGIToolGui.h
  ToolsGui/QmitkIGIToolFactory.h
  ToolsGui/QmitkIGIToolManager.h
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
)

SET(QRC_FILES
  Resources/niftkIGIGui.qrc
)