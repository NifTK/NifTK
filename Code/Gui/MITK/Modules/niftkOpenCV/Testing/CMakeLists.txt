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

if (BUILD_SLS_TRACKING_TESTING)

  mitkAddCustomModuleTest(Handeye-Sensitivity mitkHandeyeSensitivityTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Distances ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Distances_Result.txt )

  mitkAddCustomModuleTest(Tracking-Test mitkTrackingTest /home/home/thompson/Passport/SmartSurgery/TrackingTests/st/trackingtest/2013-07-17-14-03-59-348/QmitkIGINVidiaDataSource/capture-2013_7_17-13_3_59.avi)

else(BUILD_SLS_TRACKING_TESTING)

if(BUILD_SLS_TESTING)

  # Trac: 2207
  # Using OpenCV calibration, and example data from the Laparoscope, here we
  # provide a regression test for the stereo camera calibration.
  mitkAddCustomModuleTest(SLS-CameraCalib mitkCameraCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibLeft ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibRight ${CMAKE_BINARY_DIR}/Testing/Temporary/CameraCalibration 14 10 3 0.577703 0.888462)
  
endif(BUILD_SLS_TESTING)
  
mitkAddCustomModuleTest(Handeye-Calibration mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices NoSort ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_NoSort_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-DistanceSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Distances ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Distances_Result.txt )

mitkAddCustomModuleTest(Handeye-Calibration-AngleSort mitkHandeyeCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/Extrinsics.txt ${NIFTK_DATA_DIR}/Input/CameraCalibration/TrackerMatrices Angles ${NIFTK_DATA_DIR}/Input/CameraCalibration/Handeye_Angles_Result.txt )

# Trac: 2227
# Using ARUCO and OpenCV we detect AR-Tag style markers.
mitkAddCustomModuleTest(SLS-TagTracking mitkTagTrackingTest ${NIFTK_DATA_DIR}/Input/TagTracking/left.png ${NIFTK_DATA_DIR}/Input/TagTracking/right.png ${NIFTK_DATA_DIR}/Input/TagTracking/left.intrinsic.xml ${NIFTK_DATA_DIR}/Input/TagTracking/right.intrinsic.xml ${NIFTK_DATA_DIR}/Input/TagTracking/r2l.rot.xml ${NIFTK_DATA_DIR}/Input/TagTracking/r2l.trn.xml)

endif(BUILD_SLS_TRACKING_TESTING)
