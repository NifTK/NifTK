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
  mitkITKRegionParametersDataNodePropertyTest.cpp
  mitkPointUtilsTest.cpp
  mitkCoordinateAxesDataTest.cpp
  mitkCoordinateAxesDataRenderingTest.cpp
)

set(MODULE_CUSTOM_TESTS
  mitkMIDASAsAcquiredOrientationTest.cpp
  mitkMIDASCompareImagesForEqualityTest.cpp
  mitkMIDASPaintbrushToolTest.cpp
  mitkMIDASSegmentationNodeAddedVisibilityTest.cpp
  mitkMIDASMorphologicalSegmentorPipelineManagerTest.cpp
  mitkMIDASOrientationUtilsTest.cpp
  mitkMIDASImageUtilsTest.cpp
)
