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
  ToolsGui/QmitkIGIDataSourceManager.cpp
  ToolsGui/QmitkIGIDataSourceManagerClearDownThread.cpp
  ToolsGui/QmitkIGIDataSourceManagerGuiUpdateThread.cpp
)

SET(MOC_H_FILES
  ToolsGui/QmitkIGIDataSourceManager.h
)

SET(UI_FILES
  ToolsGui/QmitkIGIDataSourceManager.ui
)

SET(QRC_FILES
  Resources/niftkIGIGuiManager.qrc
)
