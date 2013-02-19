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

SET(NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES 
  niftkITK 
  ITKCommon 
  ITKIO      
  )

# This is the name of the actual executable that gets run.  
SET(MIDAS_IRREG_INTEGRATION_TESTS ${CXX_TEST_PATH}/itkMIDASIrregularVolumeEditorIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################
ADD_TEST(MIDAS-Irreg-Upd-CopyRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdateCopyRegionProcessorTest )
ADD_TEST(MIDAS-Irreg-Upd-ClearRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdateClearRegionProcessorTest )
ADD_TEST(MIDAS-Irreg-Upd-PixelWise ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdatePixelWiseSingleValueProcessorTest )
ADD_TEST(MIDAS-Irreg-RoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorTest )
ADD_TEST(MIDAS-Irreg-SliceRoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorBySlicesTest )
ADD_TEST(MIDAS-Irreg-RetainMarksNoThresh ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRetainMarksNoThresholdingTest )
ADD_TEST(MIDAS-Irreg-SplitRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest )
ADD_TEST(MIDAS-Irreg-MinROI ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorMinimumRegionTest)
ADD_TEST(MIDAS-Irreg-RegionGrowingImageFilter ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionGrowingImageFilterTest2)

#################################################################################
# Build instructions.
#################################################################################
SET( MIDASIrregIntegrationTests_SRCS
itkMIDASImageUpdateClearRegionProcessorTest.cxx
itkMIDASImageUpdateCopyRegionProcessorTest.cxx
itkMIDASImageUpdatePixelWiseSingleValueProcessorTest.cxx
itkMIDASRegionOfInterestCalculatorTest.cxx
itkMIDASRegionOfInterestCalculatorBySlicesTest.cxx
itkMIDASRetainMarksNoThresholdingTest.cxx
itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest.cxx
itkMIDASRegionOfInterestCalculatorMinimumRegionTest.cxx
itkMIDASRegionGrowingImageFilterTest2.cxx
)

ADD_EXECUTABLE(itkMIDASIrregularVolumeEditorIntegrationTests itkMIDASIrregularVolumeEditorIntegrationTests.cxx ${MIDASIrregIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(itkMIDASIrregularVolumeEditorIntegrationTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
