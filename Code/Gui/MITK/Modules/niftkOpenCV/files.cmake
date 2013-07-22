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
  Conversion/ImageConversion.cxx
  VideoProcessing/mitkBaseVideoProcessor.cxx
  VideoProcessing/mitkMonoVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoOneTimePointVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoTwoTimePointVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoDistortionCorrectionVideoProcessor.cxx
  VideoProcessing/mitkCorrectVideoFileDistortion.cxx
  VideoProcessing/mitkTrackLapUSProcessor.cxx
  VideoProcessing/mitkTrackLapUS.cxx
  CameraCalibration/mitkCameraCalibrationFacade.cxx
  CameraCalibration/mitkCameraCalibrationFromDirectory.cxx
  CameraCalibration/mitkStereoCameraCalibrationFromTwoDirectories.cxx
  CameraCalibration/mitkCorrectImageDistortion.cxx
  CameraCalibration/mitkStereoPointProjectionIntoTwoImages.cxx
  CameraCalibration/mitkHandeyeCalibrate.cxx
  CameraCalibration/Undistortion.cxx
  Registration/mitkRegistrationHelper.cxx
  Registration/mitkStereoImageToModelMetric.cxx
  Registration/mitkStereoImageToModelSSD.cxx
  Registration/mitkRegisterProbeModelToStereoPair.cxx
  Registration/mitkArunLeastSquaresPointRegistration.cxx
  Registration/mitkArunLeastSquaresPointRegistrationWrapper.cxx
  TagTracking/mitkTagTrackingFacade.cxx
  TagTracking/mitkMonoTagExtractor.cxx
  TagTracking/mitkStereoTagExtractor.cxx
  demo/mitkTestLineExtraction.cxx
  demo/mitkTestCornerExtraction.cxx
)
