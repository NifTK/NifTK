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
  Common/mitkOpenCVMaths.cxx
  Conversion/ImageConversion.cxx
  VideoProcessing/mitkBaseVideoProcessor.cxx
  VideoProcessing/mitkMonoVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoOneTimePointVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoTwoTimePointVideoProcessorTemplateMethod.cxx
  VideoProcessing/mitkStereoDistortionCorrectionVideoProcessor.cxx
  VideoProcessing/mitkCorrectVideoFileDistortion.cxx
  CameraCalibration/mitkCameraCalibrationFacade.cxx
  CameraCalibration/mitkCameraCalibrationFromDirectory.cxx
  CameraCalibration/mitkStereoCameraCalibrationFromTwoDirectories.cxx
  CameraCalibration/mitkCorrectImageDistortion.cxx
  CameraCalibration/mitkStereoPointProjectionIntoTwoImages.cxx
  CameraCalibration/mitkHandeyeCalibrate.cxx
  CameraCalibration/mitkHandeyeCalibrateFromDirectory.cxx
  CameraCalibration/Undistortion.cxx
  CameraCalibration/mitkTriangulate2DPointPairsTo3D.cxx
  Registration/mitkArunLeastSquaresPointRegistration.cxx
  Registration/mitkArunLeastSquaresPointRegistrationWrapper.cxx
  Registration/mitkLiuLeastSquaresWithNormalsRegistration.cxx
  Registration/mitkLiuLeastSquaresWithNormalsRegistrationWrapper.cxx
  TagTracking/mitkTagTrackingFacade.cxx
  TagTracking/mitkMonoTagExtractor.cxx
  TagTracking/mitkStereoTagExtractor.cxx
  VideoTrackerMatching/mitkVideoTrackerMatching.cxx
  UltrasoundCalibration/mitkUltrasoundPinCalibration.cxx
  UltrasoundCalibration/itkUltrasoundPinCalibrationCostFunction.cxx
)
