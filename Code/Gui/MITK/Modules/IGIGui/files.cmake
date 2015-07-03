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
  Common/QmitkIGIUtils.cxx
  Common/QmitkStereoImageAndCameraSelectionWidget.cxx
  Common/StereoCameraCalibrationSelectionWidget.cxx
  Common/QmitkRMSErrorWidget.cxx
  Common/QmitkUltrasoundPinCalibrationWidget.cxx
  Common/QmitkImageAndTransformSenderWidget.cxx
  Common/QmitkMatrixWidget.cxx
  Common/QmitkCalibratedModelRenderingPipeline.cxx
  OverlayEditor/QmitkBitmapOverlay.cxx
  OverlayEditor/QmitkSingle3DView.cxx
  OverlayEditor/QmitkIGIOverlayEditor.cxx
)

set(MOC_H_FILES
  Common/QmitkStereoImageAndCameraSelectionWidget.h
  Common/StereoCameraCalibrationSelectionWidget.h
  Common/QmitkRMSErrorWidget.h
  Common/QmitkUltrasoundPinCalibrationWidget.h
  Common/QmitkImageAndTransformSenderWidget.h
  Common/QmitkMatrixWidget.h
  Common/QmitkCalibratedModelRenderingPipeline.h
  OverlayEditor/QmitkSingle3DView.h
  OverlayEditor/QmitkIGIOverlayEditor.h
)

set(UI_FILES
  Common/QmitkStereoImageAndCameraSelectionWidget.ui
  Common/StereoCameraCalibrationSelectionWidget.ui
  Common/QmitkRMSErrorWidget.ui
  Common/QmitkImageAndTransformSenderWidget.ui
  Common/QmitkMatrixWidget.ui
  OverlayEditor/QmitkIGIOverlayEditor.ui
)

set(QRC_FILES
  #Resources/niftkIGIGui.qrc
)

