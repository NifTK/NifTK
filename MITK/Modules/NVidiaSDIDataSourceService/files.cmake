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
  Internal/niftkNVidiaSDIInitDialog.cxx
  Internal/niftkNVidiaSDIConfigDialog.cxx
  Internal/niftkNVidiaSDIDataSourceFactory.cxx
  Internal/niftkNVidiaSDIDataSourceActivator.cxx
  Internal/niftkNVidiaSDIDataSourceImpl.cxx
  Internal/niftkNVidiaSDIDataSourceService.cxx
  Internal/niftkNVidiaSDIDataType.cxx
)

set(MOC_H_FILES
  Internal/niftkNVidiaSDIInitDialog.h
  Internal/niftkNVidiaSDIConfigDialog.h
  Internal/niftkNVidiaSDIDataSourceImpl.h
  Internal/niftkNVidiaSDIDataSourceService.h
)

set(UI_FILES
  Internal/niftkNVidiaSDIInitDialog.ui
  Internal/niftkNVidiaSDIConfigDialog.ui
)

set(QRC_FILES
)

