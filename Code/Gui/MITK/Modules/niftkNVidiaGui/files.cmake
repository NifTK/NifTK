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
  ToolsGui/QmitkIGINVidiaDataSource.cpp 
  ToolsGui/QmitkIGINVidiaDataSourceGui.cpp 
  ToolsGui/QmitkVideoPreviewWidget.cpp
)

SET(MOC_H_FILES
  ToolsGui/QmitkIGINVidiaDataSource.h
  ToolsGui/QmitkIGINVidiaDataSourceGui.h
  ToolsGui/QmitkVideoPreviewWidget.h
)

SET(UI_FILES
  ToolsGui/QmitkIGINVidiaDataSourceGui.ui
)

SET(QRC_FILES
)