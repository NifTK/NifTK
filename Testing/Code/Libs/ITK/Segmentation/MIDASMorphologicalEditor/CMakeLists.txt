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
SET(MIDAS_MORPH_INTEGRATION_TESTS ${CXX_TEST_PATH}/MIDASMorphologicalEditorIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################
ADD_TEST(MIDAS-Morph-MeanIntensityWithinARegion ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASMeanIntensityWithinARegionFilterTest )
ADD_TEST(MIDAS-Morph-ThresholdAndAxialCutoff ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASThresholdAndAxialCutoffFilterTest )
ADD_TEST(MIDAS-Morph-ConditionalErosion ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASConditionalErosionFilterTest )
ADD_TEST(MIDAS-Morph-ConditionalDilation ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASConditionalDilationFilterTest )
ADD_TEST(MIDAS-Morph-LargestConnectedComponent ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASLargestConnectedComponentFilterTest )
ADD_TEST(MIDAS-Morph-DownSampling ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASDownSamplingFilterTest )
ADD_TEST(MIDAS-Morph-UpSampling ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASUpSamplingFilterTest )
ADD_TEST(MIDAS-Morph-Rethresholding ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/BF-MIDASRethresholding.nii.gz ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz itkMIDASRethresholdingFilterTest ${INPUT_DATA}/ADNI-AD-098_S_0149-M12-01728.img 400 ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz)
ADD_TEST(MIDAS-Morph-LimitByRegionFunction ${MIDAS_MORPH_INTEGRATION_TESTS} itkLimitByRegionFunctionTest)

# Need a new test, as filter no longer works in 2D
# ADD_TEST(MIDAS-Morph-MorphLargest ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest ${INPUT_DATA}/centered_2d 2 ${INPUT_DATA}/right_border_grey_2d 2)

#################################################################################
# Build instructions.
#################################################################################
SET( MIDASMorphIntegrationTests_SRCS
itkLimitByRegionFunctionTest.cxx
itkMIDASMeanIntensityWithinARegionFilterTest.cxx
itkMIDASThresholdAndAxialCutoffFilterTest.cxx
itkMIDASConditionalErosionFilterTest.cxx
itkMIDASConditionalDilationFilterTest.cxx
itkMIDASLargestConnectedComponentFilterTest.cxx
itkMIDASDownSamplingFilterTest.cxx
itkMIDASUpSamplingFilterTest.cxx
itkMIDASRethresholdingFilterTest.cxx
itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest.cxx
)

ADD_EXECUTABLE(MIDASMorphologicalEditorIntegrationTests MIDASMorphologicalEditorIntegrationTests ${MIDASMorphIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(MIDASMorphologicalEditorIntegrationTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
