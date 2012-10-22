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
ADD_TEST(MIDAS-Morph-ConditionalErosion ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASConditionalErosionFilterTest )
ADD_TEST(MIDAS-Morph-ConditionalDilation ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASConditionalDilationFilterTest )
ADD_TEST(MIDAS-Morph-DownSampling ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASDownSamplingFilterTest )
ADD_TEST(MIDAS-Morph-UpSampling ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASUpSamplingFilterTest )
ADD_TEST(MIDAS-Morph-Rethresholding ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/BF-MIDASRethresholding.nii.gz ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz itkMIDASRethresholdingFilterTest ${INPUT_DATA}/ADNI-AD-098_S_0149-M12-01728.img 400 ${TEMPORARY_OUTPUT}/BF-MIDASRethresholding.nii.gz)
ADD_TEST(MIDAS-Morph-LimitByRegionFunction ${MIDAS_MORPH_INTEGRATION_TESTS} itkLimitByRegionFunctionTest)
ADD_TEST(MIDAS-Morph-MaskByRegion ${MIDAS_MORPH_INTEGRATION_TESTS} itkMIDASMaskByRegionFilterTest)
ADD_TEST(MIDAS-Morph-PipeErode1 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.erode1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PE1.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PE1.nii.gz 1 59 139 96 139 1 60 160 2 0)
ADD_TEST(MIDAS-Morph-PipeDilate1 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.dilate1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PD1.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PD1.nii.gz 2 59 139 96 139 1 60 160 1 0)
ADD_TEST(MIDAS-Morph-PipeDilate2 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.dilate2.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PD1.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PD1.nii.gz 2 59 139 96 139 1 60 160 2 0)
ADD_TEST(MIDAS-Morph-PipeRethresh1 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR1.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR1.nii.gz 3 59 139 96 139 1 60 160 2 1)
ADD_TEST(MIDAS-Morph-PipeRethresh2 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh2.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR2.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR2.nii.gz 3 59 139 96 139 1 60 160 2 2)
ADD_TEST(MIDAS-Morph-PipeRethresh3 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh3.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR3.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR3.nii.gz 3 59 139 96 139 1 60 160 2 3)
ADD_TEST(MIDAS-Morph-PipeRethresh4 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh4.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR4.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR4.nii.gz 3 59 139 96 139 1 60 160 2 4)
ADD_TEST(MIDAS-Morph-PipeRethresh5 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh5.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR5.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR5.nii.gz 3 59 139 96 139 1 60 160 2 5)
ADD_TEST(MIDAS-Morph-PipeRethresh6 ${MIDAS_MORPH_INTEGRATION_TESTS} --compare ${BASELINE}/midas.rethresh6.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR6.nii.gz itkMIDASPipelineTest ${INPUT_DATA}/volunteers/16856/16856-002-1.nii.gz ${TEMPORARY_OUTPUT}/MIDAS-Morph-PR6.nii.gz 3 59 139 96 139 1 60 160 2 6)

# Need a new test, as filter no longer works in 2D
# ADD_TEST(MIDAS-Morph-MorphLargest ${SEGMENTATION_INTEGRATION_TESTS} itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest ${INPUT_DATA}/centered_2d 2 ${INPUT_DATA}/right_border_grey_2d 2)

#################################################################################
# Build instructions.
#################################################################################
SET( MIDASMorphIntegrationTests_SRCS
itkLimitByRegionFunctionTest.cxx
itkMIDASMeanIntensityWithinARegionFilterTest.cxx
itkMIDASConditionalErosionFilterTest.cxx
itkMIDASConditionalDilationFilterTest.cxx
itkMIDASDownSamplingFilterTest.cxx
itkMIDASUpSamplingFilterTest.cxx
itkMIDASRethresholdingFilterTest.cxx
itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest.cxx
itkMIDASMaskByRegionFilterTest.cxx
itkMIDASPipelineTest.cxx 
)

ADD_EXECUTABLE(MIDASMorphologicalEditorIntegrationTests MIDASMorphologicalEditorIntegrationTests ${MIDASMorphIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(MIDASMorphologicalEditorIntegrationTests ${NIFTK_TEST_EXT_ITK_SEGMENTATION_LINK_LIBRARIES} )
