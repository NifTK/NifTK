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
#  Last Changed      : $LastChangedDate: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $ 
#  Revision          : $Revision: 7447 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(NIFTK_TEST_EXT_ITK_BASIC_FILTERS_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO      
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(BASIC_FILTERS_INTEGRATION_TESTS ${CXX_TEST_PATH}/BasicFiltersIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

ADD_TEST(BF-BRescale ${BASIC_FILTERS_INTEGRATION_TESTS} BoundaryValueRescaleIntensityImageFilterTest )
ADD_TEST(BF-CPosition ${BASIC_FILTERS_INTEGRATION_TESTS} SetOutputVectorToCurrentPositionFilterTest )
ADD_TEST(BF-VecMag ${BASIC_FILTERS_INTEGRATION_TESTS} VectorMagnitudeImageFilterTest )
ADD_TEST(BF-VPlusLambdaU ${BASIC_FILTERS_INTEGRATION_TESTS} VectorVPlusLambdaUImageFilterTest )
ADD_TEST(BF-SBATest ${BASIC_FILTERS_INTEGRATION_TESTS} --compare ${BASELINE}/sba.png ${TEMPORARY_OUTPUT}/sba.png ShapeBasedAveragingImageFilterTest ${INPUT_DATA}/sba_seg1.png ${INPUT_DATA}/sba_seg2.png  ${TEMPORARY_OUTPUT}/sba.png)
ADD_TEST(BF-ExtractEdgeImage ${BASIC_FILTERS_INTEGRATION_TESTS} ExtractEdgeImageTest ${INPUT_DATA}/ADNI-AD-098_S_0149-M12-01728.img ${TEMPORARY_OUTPUT}/extract_edge.nii)
#ADD_TEST(BF-MeanCurvature ${BASIC_FILTERS_INTEGRATION_TESTS} --compare ${BASELINE}/BF-MeanCurvature_out.nii ${TEMPORARY_OUTPUT}/BF-MeanCurvature_out.nii MeanCurvatureImageFilterTest ${INPUT_DATA}/sphere_20_x_20_x_20.nii ${TEMPORARY_OUTPUT}/BF-MeanCurvature_out.nii 5 10 10 0.5)
#ADD_TEST(BF-GaussianCurvature ${BASIC_FILTERS_INTEGRATION_TESTS} GaussianCurvatureImageFilterTest ${INPUT_DATA}/sphere_20_x_20_x_20.nii ${TEMPORARY_OUTPUT}/BF-GaussianCurvature_out.nii)

#################################################################################
# Build instructions.
#################################################################################
SET( BasicFiltersIntegrationTests_SRCS
BoundaryValueRescaleIntensityImageFilterTest.cxx
SetOutputVectorToCurrentPositionFilterTest.cxx
VectorMagnitudeImageFilterTest.cxx
VectorVPlusLambdaUImageFilterTest.cxx
ShapeBasedAveragingImageFilterTest.cxx
ExtractEdgeImageTest.cxx
MeanCurvatureImageFilterTest.cxx
GaussianCurvatureImageFilterTest.cxx
)

ADD_EXECUTABLE(BasicFiltersIntegrationTests BasicFiltersIntegrationTests.cxx ${BasicFiltersIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(BasicFiltersIntegrationTests ${NIFTK_TEST_EXT_ITK_BASIC_FILTERS_LINK_LIBRARIES} )
