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
  VideoProcessing/mitkBaseVideoProcessor.cpp
  VideoProcessing/mitkMonoVideoProcessorTemplateMethod.cpp
  VideoProcessing/mitkStereoVideoProcessorTemplateMethod.cpp
  VideoProcessing/mitkStereoDistortionCorrectionVideoProcessor.cpp
  CameraCalibration/mitkCameraCalibrationFacade.cpp
  CameraCalibration/mitkCameraCalibrationFromDirectory.cpp
  CameraCalibration/mitkStereoCameraCalibrationFromTwoDirectories.cpp
  CameraCalibration/mitkCorrectImageDistortion.cpp
  CameraCalibration/mitkCorrectVideoFileDistortion.cpp
  demo/mitkOpenCVTest.cpp
)
