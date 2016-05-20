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
  mitkITKRegionParametersDataNodePropertyTest.cxx
  mitkPointUtilsTest.cxx
  mitkMergePointCloudsTest.cxx
)

set(MODULE_CUSTOM_TESTS
  niftkImageOrientationUtilsTest.cxx
  niftkAsAcquiredOrientationTest.cxx
  niftkImageUtilsTest.cxx
  niftkCompareImagesForEqualityTest.cxx
  mitkCoordinateAxesDataRenderingTest.cxx
)
