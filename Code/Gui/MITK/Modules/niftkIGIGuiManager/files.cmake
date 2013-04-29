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
  DataSources/QmitkIGIDataSourceManager.cxx
  DataSources/QmitkIGIDataSourceManagerClearDownThread.cxx
  DataSources/QmitkIGIDataSourceManagerGuiUpdateThread.cxx
)

SET(MOC_H_FILES
  DataSources/QmitkIGIDataSourceManager.h
)

SET(UI_FILES
  DataSources/QmitkIGIDataSourceManager.ui
)

SET(QRC_FILES
  Resources/niftkIGIGuiManager.qrc
)
