Notes:
------

This folder contains IGI data sources and their respective GUIs.

The QmitkIGIDataSourceManager is created within the SurgicalGuidanceView
and is responsible for instantiating sources and their GUIs.

1. Each source must be derived from QmitkIGIDataSource and not have a GUI.
2. The GUI part must be derived from QmitkIGIDataSourceGui.
3. The GUI component can have a pointer to the tool, but not the other way round.
4. The tools can use signals to broadcast information to the GUI.
5. See the .cxx file for each tool and GUI as they must define macros 
NIFTK_IGIDATASOURCE_MACRO and NIFTK_IGIDATASOURCE_GUI_MACRO for each tool.
(see QmitkIGITrackerTool.cxx and QmitkIGITrackerToolGui.cxx).
