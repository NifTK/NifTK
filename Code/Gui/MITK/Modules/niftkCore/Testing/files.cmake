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
SET(MODULE_TESTS
  mitkITKRegionParametersDataNodePropertyTest.cxx
  mitkPointUtilsTest.cxx
  mitkCoordinateAxesDataTest.cxx
)

set(MODULE_CUSTOM_TESTS
  mitkMIDASOrientationUtilsTest.cxx
  mitkMIDASAsAcquiredOrientationTest.cxx
  mitkMIDASImageUtilsTest.cxx
  mitkMIDASCompareImagesForEqualityTest.cxx
  mitkCoordinateAxesDataRenderingTest.cxx
)
