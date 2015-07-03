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
  DataSources/QmitkIGINVidiaDataSource.cxx
  DataSources/QmitkIGINVidiaDataSourceGui.cxx
  DataSources/QmitkVideoPreviewWidget.cxx
  DataSources/QmitkIGINVidiaDataSourceImpl.cxx
)

set(MOC_H_FILES
  DataSources/QmitkIGINVidiaDataSource.h
  DataSources/QmitkIGINVidiaDataSourceGui.h
  DataSources/QmitkVideoPreviewWidget.h
  DataSources/QmitkIGINVidiaDataSourceImpl.h
)

set(UI_FILES
  DataSources/QmitkIGINVidiaDataSourceGui.ui
)

set(QRC_FILES
)
