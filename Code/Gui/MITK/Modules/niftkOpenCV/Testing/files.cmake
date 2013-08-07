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

# tests with no extra command line parameter
set(MODULE_TESTS
  ImageConversionTest.cxx
)

set(MODULE_CUSTOM_TESTS
  mitkCameraCalibrationTest.cxx
  mitkHandeyeCalibrationTest.cxx
  mitkHandeyeFromDirectoryTest.cxx
  mitkTagTrackingTest.cxx
  mitkVideoTrackerMatchingTest.cxx
  mitkReprojectionTest.cxx
)
