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

SET(NIFTK_TEST_EXT_ITK_CORTICAL_THICKNESS_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(CORTICAL_THICKNESS_UNIT_TESTS ${CXX_TEST_PATH}/CorticalThicknessUnitTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

ADD_TEST(CTE-Laplace-1 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-1_out.png 255 127 0 0 10000 0.001 100 4)
ADD_TEST(CTE-Laplace-2 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-2_out.png 255 127 0 5000 7000 0.00001 10000 12)
ADD_TEST(CTE-Laplace-3 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-3_out.png 255 127 0 0 10000 0.00001 100 12)
ADD_TEST(CTE-Laplace-4 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Laplace-4_out.png ${TEMPORARY_OUTPUT}/CTE-Laplace-4_out.png LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png  ${TEMPORARY_OUTPUT}/CTE-Laplace-4_out.png 255 127 0 0 10000 0.00001 1000 321)
ADD_TEST(CTE-Laplace-5 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Laplace-5_out.png ${TEMPORARY_OUTPUT}/CTE-Laplace-5_out.png LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Laplace-5_out.png 255 127 0 0 10000 0.00001 500 150)
ADD_TEST(CTE-NormVector-1 ${CORTICAL_THICKNESS_UNIT_TESTS} ScalarImageToNormalizedGradientVectorImageFilterTest)
ADD_TEST(CTE-Stream-Int-1     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-1_out.png     255 127 0 0 10000   100 0.00001   ON 0.1    -1 -1         15  10   2.0    0.0001)
ADD_TEST(CTE-Stream-Int-2     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-2_out.png     255 127 0 0 10000   100 0.00001   ON 0.1    -1 -1         10   6   3.2    0.0001)
ADD_TEST(CTE-Stream-Int-3     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-3_out.png     255 127 0 0 10000   400 0.00001   ON 0.1    -1 -1        267 160  81.2    0.0001)
ADD_TEST(CTE-Stream-Int-4     ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Int-4_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-4_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png    ${TEMPORARY_OUTPUT}/CTE-Stream-Int-4_out.png     255 127 0 0 10000   400 0.00001   ON 0.1    -1 -1        160 297  81.2    0.0001)
ADD_TEST(CTE-Stream-Int-5     ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Int-5_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-5_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png   ${TEMPORARY_OUTPUT}/CTE-Stream-Int-5_out.png     255 127 0 0 10000 10000 0.0000001 ON 0.1    -1 -1        205 165 120.3    0.0001)
ADD_TEST(CTE-Stream-Int-6     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-6_out.png     255 127 0 0 10000  2000 0.0000001 ON 0.1    -1 -1        165 227  41.1    0.0001)
ADD_TEST(CTE-Stream-Relax-1   ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png  ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-1_out.png   255 127 0 0 10000   400 0.00001   ON -1    400 0.00001   267 160  81.0721 0.0001)
ADD_TEST(CTE-Stream-Relax-2   ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Relax-2_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-2_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png    ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-2_out.png   255 127 0 0 10000   400 0.00001   ON -1    400 0.00001   160 297  81.0611 0.0001)
ADD_TEST(CTE-Stream-Relax-3   ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Relax-3_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-3_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png   ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-3_out.png   255 127 0 0 10000 10000 0.0000001 ON -1  10000 0.0000001 205 165 120.302  0.01)
ADD_TEST(CTE-Stream-Relax-4   ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-4_out.png   255 127 0 0 10000  2000 0.0000001 ON -1   2000 0.0000001 165 227  41.0006 0.0001)
ADD_TEST(CTE-Stream-Ordered-1 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-1_out.png  255 127 0 0 1 10000 0.0000001 ON -1     -1 -1        77 288  80.9938   0.01)
ADD_TEST(CTE-Stream-Ordered-2 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-2_out.png 255 127 0 0 1 10000 0.0000001 ON -1     -1 -1         6 170 118.002    0.1)
ADD_TEST(CTE-Stream-Ordered-3 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_wiggly.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-3_out.png  255 127 0 0 1 10000 0.0000001 ON -1     -1 -1       165 227  30.3868   0.01)
ADD_TEST(CTE-Correct-PVMap ${CORTICAL_THICKNESS_UNIT_TESTS} CorrectGMUsingPVMapTest)
ADD_TEST(CTE-Correct-Neighbourhood ${CORTICAL_THICKNESS_UNIT_TESTS} CorrectGMUsingNeighbourhoodTest)
ADD_TEST(CTE-Lagrange-Init-1 ${CORTICAL_THICKNESS_UNIT_TESTS} LagrangianInitializedStreamlinesFilterTest ${TEMPORARY_OUTPUT}/CTE-Lagrange-Init-1.png ) 
ADD_TEST(CTE-Bourgeat-1 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Bourgeat-1_out.png ${TEMPORARY_OUTPUT}/CTE-Bourgeat-1_out.png Bourgeat2008Test ${INPUT_DATA}/cte_330_x_330_wiggly.png ${INPUT_DATA}/cte_330_x_330_wiggly_pv.png ${TEMPORARY_OUTPUT}/CTE-Bourgeat-1_out.png 255 127 0 200 0.000001 200 0.0000000001 0.9 0.5 0.001 2.0 153 242 59.528 0.001)
ADD_TEST(CTE-RungeKutta-1 ${CORTICAL_THICKNESS_UNIT_TESTS} FourthOrderRungeKuttaVelocityFieldTest)
ADD_TEST(CTE-RungeKutta-2 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-RungeKutta-2-out.nii ${TEMPORARY_OUTPUT}/CTE-RungeKutta-2-out.nii FourthOrderRungeKuttaVelocityFieldThicknessTest ${TEMPORARY_OUTPUT}/CTE-RungeKutta-2-out.nii)
ADD_TEST(CTE-Demons-1 ${CORTICAL_THICKNESS_UNIT_TESTS} DemonsRegistrationFilterTest)
ADD_TEST(CTE-Demons-2 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/cte_circle_update.mha ${TEMPORARY_OUTPUT}/cte_circle_update.mha DemonsRegistrationFilterUpdateTest ${INPUT_DATA}/cte_circle_wm.png ${INPUT_DATA}/cte_circle_gmwm.png ${TEMPORARY_OUTPUT}/cte_circle_update.mha)
ADD_TEST(CTE-AddUpdate-1 ${CORTICAL_THICKNESS_UNIT_TESTS} AddUpdateToTimeVaryingVelocityFilterTest )
ADD_TEST(CTE-Smooth ${CORTICAL_THICKNESS_UNIT_TESTS} GaussianSmoothVectorFieldFilterTest )
#ADD_TEST(CTE-RegCTE-1  ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/cte_circle_out.nii ${TEMPORARY_OUTPUT}/cte_circle_out.nii RegistrationBasedCorticalThicknessFilterTest ${INPUT_DATA}/cte_circle_wm.png ${INPUT_DATA}/cte_circle_gmwm.png ${INPUT_DATA}/cte_circle_thick10.png ${INPUT_DATA}/cte_circle_gwi.png 100 20 1 1.5 0 0.00001 0.99 ${TEMPORARY_OUTPUT}/cte_circle_out.nii)

#################################################################################
# Build instructions.
#################################################################################
SET( CorticalThicknessUnitTests_SRCS
LaplacianSolverImageFilterTest.cxx
ScalarImageToNormalizedGradientVectorImageFilterTest.cxx
StreamlinesFilterTest.cxx
CorrectGMUsingPVMapTest.cxx
CorrectGMUsingNeighbourhoodTest.cxx
LagrangianInitializedStreamlinesFilterTest.cxx
Bourgeat2008Test.cxx
FourthOrderRungeKuttaVelocityFieldTest.cxx
FourthOrderRungeKuttaVelocityFieldThicknessTest.cxx
DemonsRegistrationFilterTest.cxx
DemonsRegistrationFilterUpdateTest.cxx
AddUpdateToTimeVaryingVelocityFilterTest.cxx
GaussianSmoothVectorFieldFilterTest.cxx
RegistrationBasedCorticalThicknessFilterTest.cxx
)

ADD_EXECUTABLE(CorticalThicknessUnitTests CorticalThicknessUnitTests.cxx ${CorticalThicknessUnitTests_SRCS})
TARGET_LINK_LIBRARIES(CorticalThicknessUnitTests ${NIFTK_TEST_EXT_ITK_CORTICAL_THICKNESS_LINK_LIBRARIES} )
