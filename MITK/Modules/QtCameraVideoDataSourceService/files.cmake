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
  Internal/cameraframegrabber.cxx # from http://omg-it.works/how-to-grab-video-frames-directly-from-qcamera/
  Internal/niftkQtCameraDialog.cxx
  Internal/niftkQtCameraVideoDataSourceActivator.cxx
  Internal/niftkQtCameraVideoDataSourceFactory.cxx
  Internal/niftkQtCameraVideoDataSourceService.cxx
)

set(MOC_H_FILES
  Internal/cameraframegrabber.h # from http://omg-it.works/how-to-grab-video-frames-directly-from-qcamera/
  Internal/niftkQtCameraDialog.h
  Internal/niftkQtCameraVideoDataSourceService.h
)

set(UI_FILES
  Internal/niftkQtCameraDialog.ui
)

set(QRC_FILES
)
