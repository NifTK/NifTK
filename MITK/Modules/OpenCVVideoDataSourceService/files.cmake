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
  Internal/niftkOpenCVCameraDialog.cxx
  Internal/niftkOpenCVVideoDataSourceFactory.cxx
  Internal/niftkOpenCVVideoDataSourceActivator.cxx
  Internal/niftkOpenCVVideoDataSourceService.cxx
  Internal/niftkOpenCVVideoDataType.cxx
)

set(MOC_H_FILES
  Internal/niftkOpenCVCameraDialog.h
)

set(UI_FILES
  Internal/niftkOpenCVCameraDialog.ui
)

set(QRC_FILES
)

