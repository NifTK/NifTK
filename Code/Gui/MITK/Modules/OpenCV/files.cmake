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
  VideoProcessing/mitkSplitVideo.cxx
  VideoProcessing/mitkMakeGridOf2DImages.cxx
  VideoTrackerMatching/mitkProjectPointsOnStereoVideo.cxx
  VideoTrackerMatching/mitkPickPointsOnStereoVideo.cxx
  VideoTrackerMatching/mitkMakeMaskImagesFromStereoVideo.cxx
  VideoTrackerMatching/mitkFindAndTriangulateCrossHair.cxx
  VideoTrackerMatching/mitkVideoTrackerMatching.cxx
  VideoTrackerMatching/mitkTwoTrackerMatching.cxx
  VideoTrackerMatching/mitkTrackerAnalysis.cxx
  VideoTrackerMatching/mitkTwoTrackerAnalysis.cxx
  VideoTrackerMatching/mitkVideoToSurface.cxx
  CameraCalibration/mitkCameraCalibrationFacade.cxx
  CameraCalibration/mitkCameraCalibrationFromDirectory.cxx
  CameraCalibration/mitkStereoCameraCalibration.cxx
  CameraCalibration/mitkCorrectImageDistortion.cxx
  CameraCalibration/mitkStereoPointProjectionIntoTwoImages.cxx
  CameraCalibration/mitkHandeyeCalibrate.cxx
  CameraCalibration/mitkHandeyeCalibrateFromDirectory.cxx
  CameraCalibration/niftkUndistortion.cxx
  CameraCalibration/mitkTriangulate2DPointPairsTo3D.cxx
  CameraCalibration/mitkEvaluateIntrinsicParametersOnNumberOfFrames.cxx
  UltrasoundCalibration/itkInvariantPointCalibrationCostFunction.cxx
  UltrasoundCalibration/itkUltrasoundPinCalibrationCostFunction.cxx
  UltrasoundCalibration/itkVideoHandEyeCalibrationCostFunction.cxx
  UltrasoundCalibration/mitkInvariantPointCalibration.cxx
  UltrasoundCalibration/mitkUltrasoundPinCalibration.cxx
  UltrasoundCalibration/mitkVideoHandEyeCalibration.cxx
  UltrasoundCalibration/mitkUltrasoundTransformAndImageMerger.cxx
  PivotCalibration/mitkPivotCalibration.cxx
)

if(OPENCV_WITH_NONFREE)
  list(APPEND CPP_FILES Features/mitkSurfTester.cxx)
endif()
