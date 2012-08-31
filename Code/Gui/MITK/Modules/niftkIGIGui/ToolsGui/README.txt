Notes:
------

This folder contains IGI tools and their respective GUIs.

The QmitkIGIToolManager is created within the SurgicalGuidanceView
and is responsible for instantiating sockets, tools and each GUI.

1. Each tool must be derived from QmitkIGITool and not have a GUI.
2. The GUI part must be derived from QmitkIGIToolGui.
3. The GUI component can have a pointer to the tool, but not the other way round.
4. The tools can use signals to broadcast information to the GUI.
5. See the .cpp file for each tool and GUI as they must define macros 
NIFTK_IGITOOL_MACRO and NIFTK_IGITOOL_GUI_MACRO for each tool.
(see QmitkIGITrackerTool.cpp and QmitkIGITrackerToolGui.cpp).
