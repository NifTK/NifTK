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

set(NIFTK_TEST_EXT_ITK_CORTICAL_THICKNESS_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ${NIFTK_ITK_LIB}
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
set(CORTICAL_THICKNESS_UNIT_TESTS ${CXX_TEST_PATH}/CorticalThicknessUnitTests)
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

add_test(CTE-Laplace-1 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-1_out.png 255 127 0 0 10000 0.001 100 4)
add_test(CTE-Laplace-2 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-2_out.png 255 127 0 5000 7000 0.00001 10000 12)
add_test(CTE-Laplace-3 ${CORTICAL_THICKNESS_UNIT_TESTS} LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Laplace-3_out.png 255 127 0 0 10000 0.00001 100 12)
add_test(CTE-Laplace-4 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Laplace-4_out.png ${TEMPORARY_OUTPUT}/CTE-Laplace-4_out.png LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png  ${TEMPORARY_OUTPUT}/CTE-Laplace-4_out.png 255 127 0 0 10000 0.00001 1000 321)
add_test(CTE-Laplace-5 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Laplace-5_out.png ${TEMPORARY_OUTPUT}/CTE-Laplace-5_out.png LaplacianSolverImageFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Laplace-5_out.png 255 127 0 0 10000 0.00001 500 150)
add_test(CTE-NormVector-1 ${CORTICAL_THICKNESS_UNIT_TESTS} ScalarImageToNormalizedGradientVectorImageFilterTest)
add_test(CTE-Stream-Int-1     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-1_out.png     255 127 0 0 10000   100 0.00001   ON 0.1    -1 -1         15  10   2.0    0.0001)
add_test(CTE-Stream-Int-2     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_20_x_20.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-2_out.png     255 127 0 0 10000   100 0.00001   ON 0.1    -1 -1         10   6   3.2    0.0001)
add_test(CTE-Stream-Int-3     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-3_out.png     255 127 0 0 10000   400 0.00001   ON 0.1    -1 -1        267 160  81.2    0.0001)
add_test(CTE-Stream-Int-4     ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Int-4_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-4_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png    ${TEMPORARY_OUTPUT}/CTE-Stream-Int-4_out.png     255 127 0 0 10000   400 0.00001   ON 0.1    -1 -1        160 297  81.2    0.0001)
add_test(CTE-Stream-Int-5     ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Int-5_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-5_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png   ${TEMPORARY_OUTPUT}/CTE-Stream-Int-5_out.png     255 127 0 0 10000 10000 0.0000001 ON 0.1    -1 -1        205 165 120.3    0.0001)
add_test(CTE-Stream-Int-6     ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Int-6_out.png     255 127 0 0 10000  2000 0.0000001 ON 0.1    -1 -1        165 227  41.1    0.0001)
add_test(CTE-Stream-Relax-1   ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png  ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-1_out.png   255 127 0 0 10000   400 0.00001   ON -1    400 0.00001   267 160  81.0721 0.0001)
add_test(CTE-Stream-Relax-2   ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Relax-2_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-2_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png    ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-2_out.png   255 127 0 0 10000   400 0.00001   ON -1    400 0.00001   160 297  81.0611 0.0001)
add_test(CTE-Stream-Relax-3   ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Stream-Relax-3_out.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-3_out.png StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png   ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-3_out.png   255 127 0 0 10000 10000 0.0000001 ON -1  10000 0.0000001 205 165 120.302  0.01)
add_test(CTE-Stream-Relax-4   ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Relax-4_out.png   255 127 0 0 10000  2000 0.0000001 ON -1   2000 0.0000001 165 227  41.0006 0.0001)
add_test(CTE-Stream-Ordered-1 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_circle.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-1_out.png  255 127 0 0 1 10000 0.0000001 ON -1     -1 -1        77 288  80.9938   0.01)
add_test(CTE-Stream-Ordered-2 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_ellipse.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-2_out.png 255 127 0 0 1 10000 0.0000001 ON -1     -1 -1         6 170 118.002    0.1)
add_test(CTE-Stream-Ordered-3 ${CORTICAL_THICKNESS_UNIT_TESTS} StreamlinesFilterTest ${INPUT_DATA}/cte_330_x_330_wiggly.png ${TEMPORARY_OUTPUT}/CTE-Stream-Ordered-3_out.png  255 127 0 0 1 10000 0.0000001 ON -1     -1 -1       165 227  30.3868   0.01)
add_test(CTE-Correct-PVMap ${CORTICAL_THICKNESS_UNIT_TESTS} CorrectGMUsingPVMapTest)
add_test(CTE-Correct-Neighbourhood ${CORTICAL_THICKNESS_UNIT_TESTS} CorrectGMUsingNeighbourhoodTest)
add_test(CTE-Lagrange-Init-1 ${CORTICAL_THICKNESS_UNIT_TESTS} LagrangianInitializedStreamlinesFilterTest ${TEMPORARY_OUTPUT}/CTE-Lagrange-Init-1.png ) 
add_test(CTE-Bourgeat-1 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-Bourgeat-1_out.png ${TEMPORARY_OUTPUT}/CTE-Bourgeat-1_out.png Bourgeat2008Test ${INPUT_DATA}/cte_330_x_330_wiggly.png ${INPUT_DATA}/cte_330_x_330_wiggly_pv.png ${TEMPORARY_OUTPUT}/CTE-Bourgeat-1_out.png 255 127 0 200 0.000001 200 0.0000000001 0.9 0.5 0.001 2.0 153 242 59.528 0.001)
add_test(CTE-RungeKutta-1 ${CORTICAL_THICKNESS_UNIT_TESTS} FourthOrderRungeKuttaVelocityFieldTest)
add_test(CTE-RungeKutta-2 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/CTE-RungeKutta-2-out.nii ${TEMPORARY_OUTPUT}/CTE-RungeKutta-2-out.nii FourthOrderRungeKuttaVelocityFieldThicknessTest ${TEMPORARY_OUTPUT}/CTE-RungeKutta-2-out.nii)
add_test(CTE-Demons-1 ${CORTICAL_THICKNESS_UNIT_TESTS} DemonsRegistrationFilterTest)
add_test(CTE-Demons-2 ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/cte_circle_update.mha ${TEMPORARY_OUTPUT}/cte_circle_update.mha DemonsRegistrationFilterUpdateTest ${INPUT_DATA}/cte_circle_wm.png ${INPUT_DATA}/cte_circle_gmwm.png ${TEMPORARY_OUTPUT}/cte_circle_update.mha)
add_test(CTE-AddUpdate-1 ${CORTICAL_THICKNESS_UNIT_TESTS} AddUpdateToTimeVaryingVelocityFilterTest )
add_test(CTE-Smooth ${CORTICAL_THICKNESS_UNIT_TESTS} GaussianSmoothVectorFieldFilterTest )
#add_test(CTE-RegCTE-1  ${CORTICAL_THICKNESS_UNIT_TESTS} --compare ${BASELINE}/cte_circle_out.nii ${TEMPORARY_OUTPUT}/cte_circle_out.nii RegistrationBasedCorticalThicknessFilterTest ${INPUT_DATA}/cte_circle_wm.png ${INPUT_DATA}/cte_circle_gmwm.png ${INPUT_DATA}/cte_circle_thick10.png ${INPUT_DATA}/cte_circle_gwi.png 100 20 1 1.5 0 0.00001 0.99 ${TEMPORARY_OUTPUT}/cte_circle_out.nii)

#################################################################################
# Build instructions.
#################################################################################
set( CorticalThicknessUnitTests_SRCS
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

add_executable(CorticalThicknessUnitTests CorticalThicknessUnitTests.cxx ${CorticalThicknessUnitTests_SRCS})
target_link_libraries(CorticalThicknessUnitTests ${NIFTK_TEST_EXT_ITK_CORTICAL_THICKNESS_LINK_LIBRARIES} )
