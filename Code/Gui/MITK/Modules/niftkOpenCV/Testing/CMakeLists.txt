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

IF(BUILD_SLS_TESTING)

  # Trac: 2207
  # Using OpenCV calibration, and example data from the Laparoscope, here we
  # provide a regression test for the stereo camera calibration.
  mitkAddCustomModuleTest(SLS-CameraCalib mitkCameraCalibrationTest ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibLeft ${NIFTK_DATA_DIR}/Input/CameraCalibration/calibRight ${CMAKE_BINARY_DIR}/Testing/Temporary/CameraCalibration 14 10 3 0.577703 0.888462)
  
ENDIF(BUILD_SLS_TESTING)

# Trac: 2227
# Using ARUCO and OpenCV we detect AR-Tag style markers.
mitkAddCustomModuleTest(SLS-TagTracking mitkTagTrackingTest ${NIFTK_DATA_DIR}/Input/TagTracking/model.tag.regular.left-1355.png ${NIFTK_DATA_DIR}/Input/TagTracking/model.tag.regular.right-1355.png ${NIFTK_DATA_DIR}/Input/TagTracking/calib.txt.left.intrinsic.xml ${NIFTK_DATA_DIR}/Input/TagTracking/calib.txt.right.intrinsic.xml ${NIFTK_DATA_DIR}/Input/TagTracking/calib.txt.r2l.rotation.xml ${NIFTK_DATA_DIR}/Input/TagTracking/calib.txt.r2l.translation.xml)