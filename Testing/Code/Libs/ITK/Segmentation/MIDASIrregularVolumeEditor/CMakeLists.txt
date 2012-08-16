#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-10-17 14:42:00 +0100 (Mon, 17 Oct 2011) $ 
#  Revision          : $Revision: 7537 $
#  Last modified by  : $Author: sj $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES 
  niftkITK 
  ITKCommon 
  ITKIO      
  )

# This is the name of the actual executable that gets run.  
SET(MIDAS_IRREG_INTEGRATION_TESTS ${CXX_TEST_PATH}/MIDASIrregularVolumeEditorIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################
ADD_TEST(MIDAS-Irreg-Upd-CopyRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkImageUpdateCopyRegionProcessorTest )
ADD_TEST(MIDAS-Irreg-Upd-ClearRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkImageUpdateClearRegionProcessorTest )
ADD_TEST(MIDAS-Irreg-Upd-PixelWise ${MIDAS_IRREG_INTEGRATION_TESTS} itkImageUpdatePixelWiseSingleValueProcessorTest )
ADD_TEST(MIDAS-Irreg-RoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorTest )
ADD_TEST(MIDAS-Irreg-WipeSlice ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASWipeSliceTest )
ADD_TEST(MIDAS-Irreg-WipePlus ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASWipePlusTest )
ADD_TEST(MIDAS-Irreg-WipeMinus ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASWipeMinusTest )
ADD_TEST(MIDAS-Irreg-SliceRoiCalulator ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorBySlicesTest )
ADD_TEST(MIDAS-Irreg-RetainMarksNoThresh ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRetainMarksNoThresholdingTest )
ADD_TEST(MIDAS-Irreg-SplitRegion ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest )
ADD_TEST(MIDAS-Irreg-MinROI ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorMinimumRegionTest)
ADD_TEST(MIDAS-Irreg-ThreshApply ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASThresholdApplyProcessorTest)
ADD_TEST(MIDAS-Irreg-RegGrowProc ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionGrowingProcessorTest )
ADD_TEST(MIDAS-Irreg-PropUp ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASPropagateUpProcessorTest)
ADD_TEST(MIDAS-Irreg-PropDown ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASPropagateDownProcessorTest)

# itkMIDASRegionGrowingImageFilterTest is deprecated, use itkMIDASRegionGrowingImageFilterTest2 instead.
# ADD_TEST(MIDAS-Seg-RegionGrowingImageFilter ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionGrowingImageFilterTest ${INPUT_DATA}/two_regions_three_intensities.png ${INPUT_DATA}/two_regions_three_intensities_regiondef.png 1 300 ${INPUT_DATA}/two_regions_three_intensities.png ${INPUT_DATA}/two_regions_three_intensities_regiondef.png 1 200)
ADD_TEST(MIDAS-Irreg-RegionGrowingImageFilter ${MIDAS_IRREG_INTEGRATION_TESTS} itkMIDASRegionGrowingImageFilterTest2)

#################################################################################
# Build instructions.
#################################################################################
SET( MIDASIrregIntegrationTests_SRCS
itkImageUpdateClearRegionProcessorTest.cxx
itkImageUpdateCopyRegionProcessorTest.cxx
itkImageUpdatePixelWiseSingleValueProcessorTest.cxx
itkMIDASRegionOfInterestCalculatorTest.cxx
itkMIDASWipeSliceTest.cxx
itkMIDASWipePlusTest.cxx
itkMIDASWipeMinusTest.cxx
itkMIDASRegionOfInterestCalculatorBySlicesTest.cxx
itkMIDASRetainMarksNoThresholdingTest.cxx
itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest.cxx
itkMIDASRegionGrowingProcessorTest.cxx
itkMIDASPropagateUpProcessorTest.cxx
itkMIDASPropagateDownProcessorTest.cxx
itkMIDASRegionOfInterestCalculatorMinimumRegionTest.cxx
itkMIDASThresholdApplyProcessorTest.cxx
itkMIDASRegionGrowingImageFilterTest2.cxx
)

ADD_EXECUTABLE(MIDASIrregularVolumeEditorIntegrationTests MIDASIrregularVolumeEditorIntegrationTests ${MIDASIrregIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(MIDASIrregularVolumeEditorIntegrationTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
