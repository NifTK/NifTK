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

set(NIFTK_TEST_EXT_ITK_BASIC_FILTERS_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO      
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
set(BASIC_FILTERS_INTEGRATION_TESTS ${CXX_TEST_PATH}/BasicFiltersUnitTests)
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

add_test(BF-BRescale ${BASIC_FILTERS_INTEGRATION_TESTS} BoundaryValueRescaleIntensityImageFilterTest )
add_test(BF-CPosition ${BASIC_FILTERS_INTEGRATION_TESTS} SetOutputVectorToCurrentPositionFilterTest )
add_test(BF-VecMag ${BASIC_FILTERS_INTEGRATION_TESTS} VectorMagnitudeImageFilterTest )
add_test(BF-VPlusLambdaU ${BASIC_FILTERS_INTEGRATION_TESTS} VectorVPlusLambdaUImageFilterTest )
add_test(BF-SBATest ${BASIC_FILTERS_INTEGRATION_TESTS} --compare ${BASELINE}/sba.png ${TEMPORARY_OUTPUT}/sba.png ShapeBasedAveragingImageFilterTest ${INPUT_DATA}/sba_seg1.png ${INPUT_DATA}/sba_seg2.png  ${TEMPORARY_OUTPUT}/sba.png)
add_test(BF-ExtractEdgeImage ${BASIC_FILTERS_INTEGRATION_TESTS} ExtractEdgeImageTest ${INPUT_DATA}/ADNI-AD-098_S_0149-M12-01728.img ${TEMPORARY_OUTPUT}/extract_edge.nii)
#add_test(BF-MeanCurvature ${BASIC_FILTERS_INTEGRATION_TESTS} --compare ${BASELINE}/BF-MeanCurvature_out.nii ${TEMPORARY_OUTPUT}/BF-MeanCurvature_out.nii MeanCurvatureImageFilterTest ${INPUT_DATA}/sphere_20_x_20_x_20.nii ${TEMPORARY_OUTPUT}/BF-MeanCurvature_out.nii 5 10 10 0.5)
#add_test(BF-GaussianCurvature ${BASIC_FILTERS_INTEGRATION_TESTS} GaussianCurvatureImageFilterTest ${INPUT_DATA}/sphere_20_x_20_x_20.nii ${TEMPORARY_OUTPUT}/BF-GaussianCurvature_out.nii)
add_test(BF-Seg-ExcludeImageFilter ${BASIC_FILTERS_INTEGRATION_TESTS} itkExcludeImageFilterTest)
add_test(BF-LargestConnected ${BASIC_FILTERS_INTEGRATION_TESTS} itkLargestConnectedComponentFilterTest)

#################################################################################
# Build instructions.
#################################################################################
set( BasicFiltersUnitTests_SRCS
BoundaryValueRescaleIntensityImageFilterTest.cxx
SetOutputVectorToCurrentPositionFilterTest.cxx
VectorMagnitudeImageFilterTest.cxx
VectorVPlusLambdaUImageFilterTest.cxx
ShapeBasedAveragingImageFilterTest.cxx
ExtractEdgeImageTest.cxx
MeanCurvatureImageFilterTest.cxx
GaussianCurvatureImageFilterTest.cxx
itkExcludeImageFilterTest.cxx
itkLargestConnectedComponentFilterTest.cxx
)

add_executable(BasicFiltersUnitTests BasicFiltersUnitTests.cxx ${BasicFiltersUnitTests_SRCS})
target_link_libraries(BasicFiltersUnitTests ${NIFTK_TEST_EXT_ITK_BASIC_FILTERS_LINK_LIBRARIES} )
