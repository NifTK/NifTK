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
  Internal/niftkMaskMergerGUI.cxx
  Common/QmitkIGIUtils.cxx
  Common/QmitkStereoImageAndCameraSelectionWidget.cxx
  Common/StereoCameraCalibrationSelectionWidget.cxx
  Common/QmitkRMSErrorWidget.cxx
  Common/QmitkUltrasoundPinCalibrationWidget.cxx
  Common/QmitkImageAndTransformSenderWidget.cxx
  Common/QmitkMatrixWidget.cxx
  Common/QmitkCalibratedModelRenderingPipeline.cxx
  Common/QmitkVideoTestClient.cxx
  Common/QmitkVideoPreviewWidget.cxx
  MaskMerger/niftkMaskMergerController.cxx
)

set(MOC_H_FILES
  Internal/niftkMaskMergerGUI.h
  Common/QmitkStereoImageAndCameraSelectionWidget.h
  Common/StereoCameraCalibrationSelectionWidget.h
  Common/QmitkRMSErrorWidget.h
  Common/QmitkUltrasoundPinCalibrationWidget.h
  Common/QmitkImageAndTransformSenderWidget.h
  Common/QmitkMatrixWidget.h
  Common/QmitkCalibratedModelRenderingPipeline.h
  Common/QmitkVideoTestClient.h
  Common/QmitkVideoPreviewWidget.h
  MaskMerger/niftkMaskMergerController.h
)

set(UI_FILES
  Internal/niftkMaskMergerGUI.ui
  Common/QmitkStereoImageAndCameraSelectionWidget.ui
  Common/StereoCameraCalibrationSelectionWidget.ui
  Common/QmitkRMSErrorWidget.ui
  Common/QmitkImageAndTransformSenderWidget.ui
  Common/QmitkMatrixWidget.ui
)

set(QRC_FILES
  #Resources/niftkIGIGui.qrc
)

