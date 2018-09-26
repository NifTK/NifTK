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
  niftkIGIDataSourceManagerWidget.cxx
  niftkIGIDataSourcePlaybackWidget.cxx
  niftkIGIDataSourcePlaybackControlsWidget.cxx
  niftkIGIDataSourceManager.cxx
)

set(MOC_H_FILES
  niftkIGIDataSourceManagerWidget.h
  niftkIGIDataSourcePlaybackWidget.h
  niftkIGIDataSourcePlaybackControlsWidget.h
  niftkIGIDataSourceManager.h
)

set(UI_FILES
  niftkIGIDataSourceManagerWidget.ui
  niftkIGIDataSourcePlaybackWidget.ui
  niftkIGIDataSourcePlaybackControlsWidget.ui
)

set(QRC_FILES
  Resources/niftkIGIDataSourcesManager.qrc
)
