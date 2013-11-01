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

MITK_CREATE_MODULE_TESTS(LABELS niftkOpenCV)

if(BUILD_SLS_TESTING)

  # Trac: 2207
  # Using OpenCV calibration, and example data from the Laparoscope, here we
  # provide a regression test for the stereo camera calibration.
  mitkAddCustomModuleTest(SLS-CameraCalib mitkCameraCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibLeft ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibRight ${CMAKE_BINARY_DIR}/Testing/Temporary/CameraCalibration 14 10 3 0.577703 0.888462)
  
  # Trac: 2704
  # This gives different results on different platforms. Code looks OK, but optimisation just seems a
  # bit temperamental. It may be the case that the data is simply not good enough, and we need more
  # time to prove the technique. So, for the time being, turning it off, because as a regression test
  # its simply not valid.
  mitkAddCustomModuleTest(SLS-USPinCalib mitkUltrasoundPinCalibrationRegressionTest ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/matrices ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/points ${CMAKE_BINARY_DIR}/Testing/Temporary/UltrasoundPinCalibration.4x4 ${BASELINE}/UltrasoundPinCalibration.4x4 )

endif(BUILD_SLS_TESTING)
  
mitkAddCustomModuleTest(Handeye-Calibration mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices NoSort ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_NoSort_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-DistanceSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Distances ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Distances_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-AngleSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Angles ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Angles_Result.txt )

mitkAddCustomModuleTest(Video-Tracker_Matching mitkVideoTrackerMatchingTest ${NIFTK_DATA_DIR}/Input/LapTrackingAccuracy)
  
mitkAddCustomModuleTest(Reprojection-Test mitkReprojectionTest --calibration ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)

mitkAddCustomModuleTest(IdealCalibration-Test mitkIdealSteroCalibrationTest --calibration ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)

#Trac 2669 turning off the following tests as they are not very good unit tests, they take a long time to run and open up additional windows.
if(OPENCV_WITH_FFMPEG)
#  mitkAddCustomModuleTest(Handeye-from-Directory mitkHandeyeFromDirectoryTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory)

#  mitkAddCustomModuleTest(Stereo-Point-Projection mitkProjectPointsOnStereoVideoTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)

#  mitkAddCustomModuleTest(CrossHair-Triangulatrion mitkFindAndTriangulateCrossHairTest ${NIFTK_DATA_DIR}/Input/LapTrackingAccuracy/2013-07-26-17-00-36-631 ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)
else(OPENCV_WITH_FFMPEG)
# mitkAddCustomModuleTest(Handeye-from-Directory mitkHandeyeFromDirectoryTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory -NoVideoSupport)
endif(OPENCV_WITH_FFMPEG)
#end trac 2669

mitkAddCustomModuleTest(Pivot-Calibration mitkPivotCalibrationRegressionTest ${NIFTK_DATA_DIR}/Input/PivotCalibration ${CMAKE_BINARY_DIR}/Testing/Temporary/PivotCalibration.4x4 ${BASELINE}/PivotCalibration.4x4)

