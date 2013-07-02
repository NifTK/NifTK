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

set(NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES 
  niftkITK 
  ITKCommon 
  ITKIO      
  )

# This is the name of the actual executable that gets run.  
set(MIDAS_IRREG_INTEGRATION_TESTS ${CXX_TEST_PATH}/itkMIDASIrregularVolumeEditorUnitTests)
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################
add_test(MIDAS-Irreg-Upd-CopyRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdateCopyRegionProcessorTest )
add_test(MIDAS-Irreg-Upd-ClearRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdateClearRegionProcessorTest )
add_test(MIDAS-Irreg-Upd-PixelWise ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASImageUpdatePixelWiseSingleValueProcessorTest )
add_test(MIDAS-Irreg-RoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorTest )
add_test(MIDAS-Irreg-SliceRoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorBySlicesTest )
add_test(MIDAS-Irreg-RetainMarksNoThresh ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRetainMarksNoThresholdingTest )
add_test(MIDAS-Irreg-SplitRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest )
add_test(MIDAS-Irreg-MinROI ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorMinimumRegionTest)
add_test(MIDAS-Irreg-RegionGrowingImageFilter ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionGrowingImageFilterTest2)

#################################################################################
# Build instructions.
#################################################################################
set( MIDASIrregUnitTests_SRCS
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

add_executable(itkMIDASIrregularVolumeEditorUnitTests itkMIDASIrregularVolumeEditorUnitTests.cxx ${MIDASIrregUnitTests_SRCS})
target_link_libraries(itkMIDASIrregularVolumeEditorUnitTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
