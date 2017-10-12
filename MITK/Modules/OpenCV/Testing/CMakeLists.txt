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

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Suppress warning because of a bug in the Boost gil library.
  set_target_properties(${TESTDRIVER} PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
endif()

mitk_use_modules(TARGET ${TESTDRIVER}
  MODULES niftkcommon niftkVTK
  PACKAGES OpenCV
)

mitkAddCustomModuleTest(CameraCalibRegression mitkCameraCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibLeft ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibRight ${CMAKE_BINARY_DIR}/Testing/Temporary/CameraCalibration 14 10 3 0.577703 0.913)

mitkAddCustomModuleTest(USPinCalibRegression mitkUltrasoundPinCalibrationRegressionTest ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/2015.04.23-ultrasound_calibration/Aurora_2/1/ ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/2015.04.23-ultrasound_calibration/pins/ ${CMAKE_BINARY_DIR}/Testing/Temporary/UltrasoundPinCalibration.4x4 ${BASELINE}/UltrasoundPinCalibration.4x4  ${NIFTK_DATA_DIR}/Input/UltrasoundPinCalibration/2015.04.23-ultrasound_calibration/initialParameters.txt)
  
mitkAddCustomModuleTest(Handeye-Calibration mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices NoSort ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_NoSort_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-DistanceSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Distances ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Distances_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-AngleSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Angles ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Angles_Result.txt )

mitkAddCustomModuleTest(Video-Tracker_Matching mitkVideoTrackerMatchingTest ${NIFTK_DATA_DIR}/Input/LapTrackingAccuracy)
  
mitkAddCustomModuleTest(Reprojection-Test mitkReprojectionTest --calibration ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)

mitkAddCustomModuleTest(UndistortionLoop-Test mitkUndistortionLoopTest --calibration ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)

mitkAddCustomModuleTest(IdealCalibration-Test mitkIdealStereoCalibrationTest --calibration ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration --tracking ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration/TrackerMatrices0 --gridToWorldTransform ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration/calib.left.gridtoworld.txt --maxTrackingMatricesToUse 6)

mitkAddCustomModuleTest(Stereo-Point-Projection mitkProjectPointsOnStereoVideoTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration ${TEMP})

mitkAddCustomModuleTest(Project-Rays-Test mitkProjectCameraRaysTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices/1359466034091194000.txt)

#Trac 2669 turning off the following tests as they are not very good unit tests, they take a long time to run and open up additional windows.
if(OPENCV_WITH_FFMPEG)
#  mitkAddCustomModuleTest(Handeye-from-Directory mitkHandeyeFromDirectoryTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory)


#  mitkAddCustomModuleTest(CrossHair-Triangulatrion mitkFindAndTriangulateCrossHairTest ${NIFTK_DATA_DIR}/Input/LapTrackingAccuracy/2013-07-26-17-00-36-631 ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory/CertusCalibration)
else(OPENCV_WITH_FFMPEG)
# mitkAddCustomModuleTest(Handeye-from-Directory mitkHandeyeFromDirectoryTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/HandeyeFromDirectory -NoVideoSupport)
endif(OPENCV_WITH_FFMPEG)
#end trac 2669

mitkAddCustomModuleTest(Pivot-Calibration mitkPivotCalibrationRegressionTest ${NIFTK_DATA_DIR}/Input/PivotCalibration ${CMAKE_BINARY_DIR}/Testing/Temporary/PivotCalibration.4x4 ${BASELINE}/PivotCalibration.4x4)

if(OPENCV_WITH_NONFREE)
  mitkAddCustomModuleTest(DistanceFromCamera1 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822838538400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822838538400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 30.898)
  mitkAddCustomModuleTest(DistanceFromCamera2 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822903544800_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822903544800_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 31.495)
  mitkAddCustomModuleTest(DistanceFromCamera3 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822918546400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822918546400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 32.176)
  mitkAddCustomModuleTest(DistanceFromCamera4 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822960550400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822960550400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 35.226)
  mitkAddCustomModuleTest(DistanceFromCamera5 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822998554400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900822998554400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 33.523)
  mitkAddCustomModuleTest(DistanceFromCamera6 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900823038558400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900823038558400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 34.087)
  mitkAddCustomModuleTest(DistanceFromCamera7 niftkDistanceFromCameraTest ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900823078562400_left.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/1432900823078562400_right.bmp ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.left.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.right.intrinsic.txt ${NIFTK_DATA_DIR}/Input/DistanceMeasurer/calib.r2l.txt 33.458)
endif()
