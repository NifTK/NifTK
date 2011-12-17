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
SET(SEGMENTATION_INTEGRATION_TESTS ${CXX_TEST_PATH}/SegmentationIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)
SET(BASELINE ${NIFTK_DATA_ROOT}/Baseline)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################
ADD_TEST(MIDAS-Seg-MeanIntensityWithinARegion ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASMeanIntensityWithinARegionFilterTest )
ADD_TEST(MIDAS-Seg-ThresholdAndAxialCutoff ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASThresholdAndAxialCutoffFilterTest )
ADD_TEST(MIDAS-Seg-ConditionalErosion ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASConditionalErosionFilterTest )
ADD_TEST(MIDAS-Seg-ConditionalDilation ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASConditionalDilationFilterTest )
ADD_TEST(MIDAS-Seg-LargestConnectedComponent ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASLargestConnectedComponentFilterTest )
ADD_TEST(MIDAS-Seg-DownSampling ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASDownSamplingFilterTest )
ADD_TEST(MIDAS-Seg-UpSampling ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASUpSamplingFilterTest )
ADD_TEST(MIDAS-Seg-Rethresholding ${SEGMENTATION_INTEGRATION_TESTS} --compare ${BASELINE}/BF-MIDASRethresholding.nii.gz ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz itkMIDASRethresholdingFilterTest ${INPUT_DATA}/ADNI-AD-098_S_0149-M12-01728.img 400 ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz)
ADD_TEST(MIDAS-Seg-ExcludeImageFilter ${SEGMENTATION_INTEGRATION_TESTS} itkExcludeImageFilterTest)
ADD_TEST(MIDAS-Seg-LimitByRegionFunction ${SEGMENTATION_INTEGRATION_TESTS} itkLimitByRegionFunctionTest)
ADD_TEST(MIDAS-Seg-RegionGrowingImageFilter ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionGrowingImageFilterTest ${INPUT_DATA}/two_regions_three_intensities.png ${INPUT_DATA}/two_regions_three_intensities_regiondef.png 1 300 ${INPUT_DATA}/two_regions_three_intensities.png ${INPUT_DATA}/two_regions_three_intensities_regiondef.png 1 200)
ADD_TEST(MIDAS-Seg-Upd-CopyRegion ${SEGMENTATION_INTEGRATION_TESTS} itkImageUpdateCopyRegionProcessorTest )
ADD_TEST(MIDAS-Seg-Upd-ClearRegion ${SEGMENTATION_INTEGRATION_TESTS} itkImageUpdateClearRegionProcessorTest )
ADD_TEST(MIDAS-Seg-Upd-PixelWise ${SEGMENTATION_INTEGRATION_TESTS} itkImageUpdatePixelWiseSingleValueProcessorTest )
ADD_TEST(MIDAS-Seg-RoiCalulator ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorTest )
ADD_TEST(MIDAS-Seg-WipeSlice ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASWipeSliceTest )
ADD_TEST(MIDAS-Seg-WipePlus ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASWipePlusTest )
ADD_TEST(MIDAS-Seg-WipeMinus ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASWipeMinusTest )
ADD_TEST(MIDAS-Seg-SliceRoiCalulator ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorBySlicesTest )
ADD_TEST(MIDAS-Seg-RetainMarksNoThresh ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRetainMarksNoThresholdingTest )
ADD_TEST(MIDAS-Seg-SplitRegion ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest )
ADD_TEST(MIDAS-Seg-RegGrowProc ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionGrowingProcessorTest )
ADD_TEST(MIDAS-Seg-PropUp ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASPropagateUpProcessorTest)
ADD_TEST(MIDAS-Seg-PropDown ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASPropagateDownProcessorTest)
ADD_TEST(MIDAS-Seg-MinROI ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASRegionOfInterestCalculatorMinimumRegionTest)
ADD_TEST(MIDAS-Seg-ThreshApply ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASThresholdApplyProcessorTest)
ADD_TEST(MIDAS-Seg-MorphLargest ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest ${INPUT_DATA}/centered_2d 2 ${INPUT_DATA}/right_border_grey_2d 2)

#################################################################################
# Build instructions.
#################################################################################
SET( SegmentationIntegrationTests_SRCS
itkExcludeImageFilterTest.cxx
itkLimitByRegionFunctionTest.cxx
itkImageUpdateClearRegionProcessorTest.cxx
itkImageUpdateCopyRegionProcessorTest.cxx
itkImageUpdatePixelWiseSingleValueProcessorTest.cxx
itkMIDASMeanIntensityWithinARegionFilterTest.cxx
itkMIDASThresholdAndAxialCutoffFilterTest.cxx
itkMIDASConditionalErosionFilterTest.cxx
itkMIDASConditionalDilationFilterTest.cxx
itkMIDASLargestConnectedComponentFilterTest.cxx
itkMIDASDownSamplingFilterTest.cxx
itkMIDASUpSamplingFilterTest.cxx
itkMIDASRethresholdingFilterTest.cxx
itkMIDASRegionGrowingImageFilterTest.cxx
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
itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest.cxx
)

ADD_EXECUTABLE(SegmentationIntegrationTests SegmentationIntegrationTests ${SegmentationIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(SegmentationIntegrationTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
