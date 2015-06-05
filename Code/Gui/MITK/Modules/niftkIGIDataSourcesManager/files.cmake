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
  QmitkIGIDataSourceManager.cxx
  QmitkIGIDataSourceManagerClearDownThread.cxx
  QmitkIGIDataSourceManagerGuiUpdateThread.cxx
)

set(MOC_H_FILES
  QmitkIGIDataSourceManager.h
)

set(UI_FILES
  QmitkIGIDataSourceManager.ui
)

set(QRC_FILES
  Resources/niftkIGIDataSourcesManager.qrc
)
