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

SET(NIFTK_TEST_EXT_ITK_REGISTRATION_TOOLBOX_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO
  ITKStatistics
  ITKAlgorithms  
  ${Boost_LIBRARIES}
  )

# These are the names of the actual executable that gets run.  
SET(REGISTRATION_TOOLBOX_INTEGRATION_TESTS ${CXX_TEST_PATH}/RegistrationToolboxUnitTests)

# This is where we should dump any output files.
SET(TEMP_DIR ${NIFTK_BINARY_DIR}/Testing/Temporary )

#----------------------------------------------------------------------------------------------------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#----------------------------------------------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

# These tests whether the Switchable affine transform is switching the number of DOF correctly at 2D and 3D.
ADD_TEST(Switch-Affine-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SwitchableAffineTransformTest)
ADD_TEST(Switch-Affine-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SwitchableAffine2DTransformTest)

# 2D coordinate tests. The columns are...............................................           cx cy rz tx  ty sx  sy  ipx  ipy exx exy 
ADD_TEST(Affine2D-Coords-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0  0   0  1   1   0    0   0   0   )
ADD_TEST(Affine2D-Coords-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  90 0   0  1   1   1    0   0   -1  )
ADD_TEST(Affine2D-Coords-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0  1   0  1   1   1    2   2   2   )
ADD_TEST(Affine2D-Coords-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0 -1   0  1   1   2    3   1   3   )
ADD_TEST(Affine2D-Coords-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0  0   1  1   1   3    4   3   5   )
ADD_TEST(Affine2D-Coords-6 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0  0  -1  1   1   4    5   4   4   )
ADD_TEST(Affine2D-Coords-7 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  0  0  0  0   0  1.1 1.1 1    2   1.1 2.2 )

# If you had an image 200, 150 with 1.5, 1.4 mm voxels, and the origin in ITK terms (top left of image), 
# is (0,0,0) and you wanted to rotate about the centre, the centre is: 149.25, 104.3
ADD_TEST(Affine2D-Coords-7 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  149.25 104.3  0   0  0  1  1  149.25 104.3 149.25 104.3)
ADD_TEST(Affine2D-Coords-8 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest  149.25 104.3  90  0  0  1  1  150.25 104.3 149.25 103.3)
ADD_TEST(Affine2D-Coords-9 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DTransformTest 149.25 104.3  90  0  0  1  1  149.25 105.3 150.25 104.3)

# 3D coordinate tests. The columns are............................................................. cx cy cz rx  ry rz tx ty tz sx   sy   sz  ipx  ipy ipz exx exy exz
ADD_TEST(Affine3D-Coords-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0   0  0  0  0  0  1    1    1   0    0   0   0   0   0   )
ADD_TEST(Affine3D-Coords-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  90  0  0  0  0  0  1    1    1   1    0   0   1   0   0   )
ADD_TEST(Affine3D-Coords-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  90  0  0  0  0  0  1    1    1   0    1   0   0   0   -1  )
ADD_TEST(Affine3D-Coords-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  -90 0  0  0  0  0  1    1    1   0    1   0   0   0   1   )
ADD_TEST(Affine3D-Coords-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0  90  0  0  0  0  1    1    1   0    1   0   0   1   0   )
ADD_TEST(Affine3D-Coords-6 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0  90  0  0  0  0  1    1    1   0    0   1   1   0   0   )
ADD_TEST(Affine3D-Coords-7 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0 -90  0  0  0  0  1    1    1   0    0   1  -1   0   0   )
ADD_TEST(Affine3D-Coords-8 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0   0  0  1  0  0  1    1    1   1    2   3   2   2   3   )
ADD_TEST(Affine3D-Coords-9 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest  0  0  0  0   0  0  0  1  0  1    1    1   1    2   3   1   3   3   )
ADD_TEST(Affine3D-Coords-10 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 0  0  0  0   0  0  0  0  1  1    1    1   1    2   3   1   2   4   )
ADD_TEST(Affine3D-Coords-11 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 0  0  0  0   0  0  0  0  0  1.1  1    1   1    2   3   1.1 2   3   )
ADD_TEST(Affine3D-Coords-12 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 0  0  0  0   0  0  0  0  0  1    1.1  1   1    2   3   1   2.2 3   )
ADD_TEST(Affine3D-Coords-13 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 0  0  0  0   0  0  0  0  0  1    1    1.1 1    2   3   1   2   3.3 )

# If you had an image 200, 150, 100 with 1.5, 1.4, 1.3 mm voxels, and the origin in ITK terms (top left of image), 
# is (0,0,0) and you wanted to rotate about the centre, the centre is: 149.25, 104.3, 64.35
ADD_TEST(Affine3D-Coords-14 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 149.25 104.3  64.35 0   0  0  0  0  0  1  1  1  150.25 104.3 64.35 150.25 104.3 64.35   )
ADD_TEST(Affine3D-Coords-15 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 149.25 104.3  64.35 90  0  0  0  0  0  1  1  1  150.25 104.3 64.35 150.25 104.3 64.35   )
ADD_TEST(Affine3D-Coords-16 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DTransformTest 149.25 104.3  64.35 90  0  0  0  0  0  1  1  1  149.25 105.3 64.35 149.25 104.3 63.35   )

# One long script to test 2D jacobian. Not really sure how to test it, so this test needs expanding.
ADD_TEST(Affine2DJacobianTest ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine2DJacobianTest)

# One long script to test 3D jacobian. Not really sure how to test it, so this test needs expanding.
ADD_TEST(Affine3DJacobianTest ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} EulerAffine3DJacobianTest)

# The columns are............................................................. cx cy cz rx  ry rz tx ty tz sx   sy   sz 
ADD_TEST(MatrixLinearCombinationFunctionsTests-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MatrixLinearCombinationFunctionsTests  0  0  0  30   0  0  1  1  1  1   1  1)
ADD_TEST(MatrixLinearCombinationFunctionsTests-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MatrixLinearCombinationFunctionsTests  0  0  0  0   60  0  1  1  1  1   1  1)
ADD_TEST(MatrixLinearCombinationFunctionsTests-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MatrixLinearCombinationFunctionsTests  0  0  0  0   0  90  1  1  1  1   1  1)

# BSpline transform tests.
#ADD_TEST(BSpline-2D-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} BSplineTransformTest ${INPUT_DATA}/fluid_fixed_10_x_10.png 2 2 6 6 1 10 3.74394 0.0 1.90744 -0.657072 ${TEMP_DIR}/BSplineTransformTest_10.png )
ADD_TEST(BSpline-2D-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} BSplineTransformTest ${INPUT_DATA}/grid.png 10 10 26 26 10 10 4.43353 0.0 1.42974 0.570887 ${TEMP_DIR}/BSplineTransformTest_grid.png )

#################################################################################
# Now test the metrics.
#################################################################################

# Origin
ADD_TEST(Metric-SSD-1  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0 30)

# Shift X
ADD_TEST(Metric-SSD-2  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 192544 24 )
ADD_TEST(Metric-SSD-3  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 192544 24 )

# Shift Y
ADD_TEST(Metric-SSD-4  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 225304 20 )
ADD_TEST(Metric-SSD-5  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 225304 20 )

# Test Fixed lower threshold at origin
ADD_TEST(Metric-SSD-6  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 10 255 0 255 0 12)

# Test Fixed upper threshold at origin
ADD_TEST(Metric-SSD-7  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 130 0 255 0 28)

# Test moving lower threshold at origin
ADD_TEST(Metric-SSD-8  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 10 255 0 12)

# Test moving upper threshold at origin
ADD_TEST(Metric-SSD-9  ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 130 0 28)

# Same as last 4, except with translation of 1.
ADD_TEST(Metric-SSD-10 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 10 255 0 255 129040 12)
ADD_TEST(Metric-SSD-11 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 130 0 255 159776 22)
ADD_TEST(Metric-SSD-12 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 10 255 129040 12)
ADD_TEST(Metric-SSD-13 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 130 159776 22)

# Different image.
ADD_TEST(Metric-SSD-14 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey2.png 0 0 0 255 0 255 6050 12)
ADD_TEST(Metric-SSD-15 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 1 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey2.png 1 0 0 255 0 255 106930 12)

# Sum Absolute Difference
ADD_TEST(Metric-MSD-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 2 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0 30 )
ADD_TEST(Metric-MSD-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 2 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 8022.666666666667 24 )
ADD_TEST(Metric-MSD-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 2 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 8022.666666666667 24 )
ADD_TEST(Metric-MSD-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 2 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 11265.20000000000 20 )
ADD_TEST(Metric-MSD-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 2 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 11265.20000000000 20 )

# Sum Absolute Difference
ADD_TEST(Metric-SAD-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 3 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0 30 )
ADD_TEST(Metric-SAD-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 3 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 1520 24 )
ADD_TEST(Metric-SAD-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 3 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 1520 24 )
ADD_TEST(Metric-SAD-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 3 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 1268 20 )
ADD_TEST(Metric-SAD-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 3 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 1268 20 )

# Normalized cross correlation
ADD_TEST(Metric-NCC-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 4 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 1 30 )
ADD_TEST(Metric-NCC-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 4 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 0.151275 24 )
ADD_TEST(Metric-NCC-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 4 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 0.151275 24 )
ADD_TEST(Metric-NCC-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 4 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 0.0463503 20 )
ADD_TEST(Metric-NCC-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 4 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 0.0463503 20)

# Ratio Image Uniformity. RIU is a term used in Hill/Hawkes/Hajnal book. Its Roger Woods JCAT 92 original PET-PET measure
ADD_TEST(Metric-RIU-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 5 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0 30 )
ADD_TEST(Metric-RIU-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 5 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 2.195311039 24 )
ADD_TEST(Metric-RIU-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 5 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 2.195311039 24 )
ADD_TEST(Metric-RIU-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 5 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 2.533998124 20 )
ADD_TEST(Metric-RIU-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 5 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 2.533998124 20)

# Partitioned Image Uniformity. PIU is a term used in Hill/Hawkes/Hajnal book. Its Roger Woods JCAT 93 original MR-PET measure
# but later papers seem to use RIU more.
ADD_TEST(Metric-PIU-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 6 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0 30 )
ADD_TEST(Metric-PIU-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 6 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 1.070698118 24 )
ADD_TEST(Metric-PIU-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 6 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 1.070698118 24 )
ADD_TEST(Metric-PIU-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 6 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 1.347094930 20)
ADD_TEST(Metric-PIU-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 6 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 1.347094930 20)

# Joint Entropy.
ADD_TEST(Metric-JE-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 7 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0.8532361666 30 )
ADD_TEST(Metric-JE-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 7 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 1.676234960 24 )
ADD_TEST(Metric-JE-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 7 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 1.676234960 24 )
ADD_TEST(Metric-JE-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 7 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255  1.748067378997 20)
ADD_TEST(Metric-JE-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 7 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 1.748067378997 20)

# Mutual Information
ADD_TEST(Metric-MI-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 8 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 0.853236166 30 )
ADD_TEST(Metric-MI-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 8 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 0.1606206099 24 )
ADD_TEST(Metric-MI-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 8 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 0.1606206099 24 )
ADD_TEST(Metric-MI-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 8 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 0.1049458622932 20)
ADD_TEST(Metric-MI-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 8 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 0.1049458622932 20)

# Normalized Mutual Information
ADD_TEST(Metric-NMI-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 9 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 2 30 )
ADD_TEST(Metric-NMI-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 9 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 1.095822252 24 )
ADD_TEST(Metric-NMI-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 9 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 1.095822252 24 )
ADD_TEST(Metric-NMI-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 9 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 1.060035364514 20)
ADD_TEST(Metric-NMI-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 9 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 1.060035364514 20)

# Correlation Ratio
ADD_TEST(Metric-CR-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 10 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 0 0 255 0 255 1 30 )
ADD_TEST(Metric-CR-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 10 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 1 0 0 255 0 255 0.542556 24 )
ADD_TEST(Metric-CR-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 10 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png -1 0 0 255 0 255 0.542556 24 )
ADD_TEST(Metric-CR-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 10 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 2 0 255 0 255 0.474778 20)
ADD_TEST(Metric-CR-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageMetricTest2D 10 ${INPUT_DATA}/5By6Grey.png ${INPUT_DATA}/5By6Grey.png 0 -2 0 255 0 255 0.474778 20)

#################################################################################
# Pure Optimizer tests. These are really so we can make sure we understand what the
# optimizers actually do, and whether the parameters work. 
# Also, it works as a regression test.
#################################################################################

ADD_TEST(Simplex-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    1      0.678 0.678 OFF 100 1.00 -0.17        -0.17)
ADD_TEST(Simplex-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 0.1  1      0.678 0.678 OFF 100 1.00  0.0192184    0.0042796)
ADD_TEST(Simplex-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 0.01 1      0.678 0.678 OFF 100 1.00  0.00718215  -0.00245077)
ADD_TEST(Simplex-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    0.1    0.678 0.678 OFF 100 1.00 -0.17        -0.17)
ADD_TEST(Simplex-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    0.01   0.678 0.678 OFF 100 1.00 -0.17        -0.17)
ADD_TEST(Simplex-6 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    0.001  0.678 0.678 OFF 100 1.00 -0.000696564 -0.0185114)
ADD_TEST(Simplex-7 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    0.0001 0.678 0.678 OFF 100 1.00  0.00718215  -0.00245077)
ADD_TEST(Simplex-8 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLSimplexTest 10 10 1    0.0001 0.678 0.678 OFF 10  1.00  5.593        4.915)

ADD_TEST(RegularStep-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 1       2.34728 0.5   0.1  100  0.001 -0.214856   -0.233197)
ADD_TEST(RegularStep-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 1       2.34728 0.005 0.1  100  0.001  0.580395    0.629941)
ADD_TEST(RegularStep-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 0.001   2.34728 0.5   0.1  100  0.001 -0.0160429  -0.0174125)
ADD_TEST(RegularStep-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 0.00001 2.34728 0.5   0.1  100  0.001 -0.0160429  -0.0174125)
ADD_TEST(RegularStep-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 0.00001 2.34728 0.5   0.01 100  0.001  0.00259574  0.00281733)
ADD_TEST(RegularStep-6 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLRegularStepOptimizerTest 10.1234 10.9876 0.00001 2.34728 0.5   0.01   5  0.001  3.7614      4.08249)

ADD_TEST(GradientDescent-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLGradientDescentOptimizerTest 10.1234 10.9876 1   10   0.00001 10.1234      10.9876)
ADD_TEST(GradientDescent-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLGradientDescentOptimizerTest 10.1234 10.9876 0.9 10   0.00001  1.08699      1.17978)
ADD_TEST(GradientDescent-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLGradientDescentOptimizerTest 10.1234 10.9876 0.8 10   0.00001  0.0612123    0.0664378)
ADD_TEST(GradientDescent-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLGradientDescentOptimizerTest 10.1234 10.9876 0.1 10   0.00001  1.08699      1.17978)
ADD_TEST(GradientDescent-5 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SquaredUCLGradientDescentOptimizerTest 10.1234 10.9876 0.1 100  0.00001  2.06217e-09  2.23821e-09)

#################################################################################
# Block Matching Test. Leave these out, as the debugging output makes
# it take a prohibitively long time.
#################################################################################

#ADD_TEST(Block-2D-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DBlockMatchingTest ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png ${TEMP_DIR}/block_2d_1_out.png 5 5 2 0 0 0 100 0.1)

#################################################################################
# Deformable stuff.
#################################################################################

ADD_TEST(Fluid-NMI-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} NMILocalHistogramDerivativeForceFilterTest ${INPUT_DATA}/fluid_fixed_10_x_10.png ${INPUT_DATA}/fluid_fixed_10_x_10_1_pixel_diff.png 1.7664673516718401 0.0001 4 2 0 -0.00503801 )
ADD_TEST(Fluid-NMI-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} NMILocalHistogramDerivativeForceFilterTest ${INPUT_DATA}/fluid_fixed_10_x_10.png ${INPUT_DATA}/fluid_fixed_10_x_10_1_pixel_diff.png 1.7664673516718401 0.0001 6 7 0 0.00503801 )
ADD_TEST(Force-Parzen-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} HistogramParzenWindowDerivativeForceFilterTest)
ADD_TEST(Force-Hist-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} itkHistogramRegistrationForceGeneratorTest)

ADD_TEST(BSpline-Smooth-1      ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} BSplineSmoothTest )
ADD_TEST(BSpline-Interpolate-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} BSplineInterpolateTest )

################################################################################################
# Now test the overall registration algorithms. Leave these at the end, as they take the longest
################################################################################################

ADD_TEST(Reg2D-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMeanSquaresTest ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png  5  5 0 0 0.0001)
ADD_TEST(Reg2D-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMeanSquaresTest ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png -5 -5 0 0 0.0001)
ADD_TEST(Reg2D-Multi-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MultiRes2DMeanSquaresTest 1 ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png 5 5 0 0)
ADD_TEST(Reg2D-Multi-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MultiRes2DMeanSquaresTest 1 ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png -5 -5 0 0)
ADD_TEST(Reg2D-Multi-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MultiRes2DMeanSquaresTest 2 ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png 5 5 0 0)
ADD_TEST(Reg2D-Multi-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} MultiRes2DMeanSquaresTest 2 ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png -5 -5 0 0)
ADD_TEST(Reg2D-Mask-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DCorrelationMaskTest ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii  1 0 0 0 0 0 0.0001)
ADD_TEST(Reg2D-Mask-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DCorrelationMaskTest OFF ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii 1 0 0 0 0 0 0.0001)
ADD_TEST(Reg2D-Mask-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DCorrelationMaskTest ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii 3 3 0 0 0 0 0.0001)

#ADD_TEST(Reg2D-TaR-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMultiStageMethodTest 2 ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii  0 0 0 0 0 0)
#ADD_TEST(Reg2D-TaR-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMultiStageMethodTest 3 ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii  0 0 0 0 0 0)
#ADD_TEST(Reg2D-TaR-3 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMultiStageMethodTest 4 ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii  0 0 0 0 0 0)
#ADD_TEST(Reg2D-TaR-4 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SingleRes2DMultiStageMethodTest 2 ON ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle-x1-y1-r1.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle_REG.nii ${INPUT_DATA}/ADNI-C-052_S_1250-Scr-01873-middle-x1-y1-r1_REG.nii  0 0 0 0 0 0)

# Filter. The aim is just to test that the filter can launch the registration and resample.
ADD_TEST(RegFilter1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} ImageRegistrationFilterTest ${INPUT_DATA}/BrainProtonDensitySlice.png ${INPUT_DATA}/BrainProtonDensitySlice.png)

#################################################################################
# Deformable stuff.
#################################################################################
#                                                                                                                                                                         scaleByGradient    scaleComponentWise constraintGradient parzenWindows bins weighting                 
ADD_TEST(FFD-Reg-Circle-01 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 FALSE              FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_1.png ${TEMP_DIR}/ffd_register_grid_1.png ${TEMP_DIR}/ffd_register_after_diff_1.png ${TEMP_DIR}/ffd_register_before_diff_1.png ${TEMP_DIR}/ffd_register_op_1.txt ${BASELINE}/ffd_register_op_1.txt)
#ADD_TEST(FFD-Reg-Circle-02 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             10 10 1  2 FALSE              FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_2.png ${TEMP_DIR}/ffd_register_grid_2.png ${TEMP_DIR}/ffd_register_after_diff_2.png ${TEMP_DIR}/ffd_register_before_diff_2.png ${TEMP_DIR}/ffd_register_op_2.txt ${BASELINE}/ffd_register_op_2.txt)
#ADD_TEST(FFD-Reg-Circle-03 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png              5  5 1  3 FALSE              FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_3.png ${TEMP_DIR}/ffd_register_grid_3.png ${TEMP_DIR}/ffd_register_after_diff_3.png ${TEMP_DIR}/ffd_register_before_diff_3.png ${TEMP_DIR}/ffd_register_op_3.txt ${BASELINE}/ffd_register_op_3.txt)
ADD_TEST(FFD-Reg-Circle-04 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 TRUE               TRUE               FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_4.png ${TEMP_DIR}/ffd_register_grid_4.png ${TEMP_DIR}/ffd_register_after_diff_4.png ${TEMP_DIR}/ffd_register_before_diff_4.png ${TEMP_DIR}/ffd_register_op_4.txt ${BASELINE}/ffd_register_op_4.txt)
ADD_TEST(FFD-Reg-Circle-05 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 TRUE               FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_5.png ${TEMP_DIR}/ffd_register_grid_5.png ${TEMP_DIR}/ffd_register_after_diff_5.png ${TEMP_DIR}/ffd_register_before_diff_5.png ${TEMP_DIR}/ffd_register_op_5.txt ${BASELINE}/ffd_register_op_5.txt)
ADD_TEST(FFD-Reg-Circle-06 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 FALSE              FALSE              TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_6.png ${TEMP_DIR}/ffd_register_grid_6.png ${TEMP_DIR}/ffd_register_after_diff_6.png ${TEMP_DIR}/ffd_register_before_diff_6.png ${TEMP_DIR}/ffd_register_op_6.txt ${BASELINE}/ffd_register_op_6.txt)
#ADD_TEST(FFD-Reg-Circle-07 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 TRUE               TRUE               TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_7.png ${TEMP_DIR}/ffd_register_grid_7.png ${TEMP_DIR}/ffd_register_after_diff_7.png ${TEMP_DIR}/ffd_register_before_diff_7.png ${TEMP_DIR}/ffd_register_op_7.txt ${BASELINE}/ffd_register_op_7.txt)
ADD_TEST(FFD-Reg-Circle-08 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} FFDRegisterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png             15 15 10 1 TRUE               FALSE              TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_8.png ${TEMP_DIR}/ffd_register_grid_8.png ${TEMP_DIR}/ffd_register_after_diff_8.png ${TEMP_DIR}/ffd_register_before_diff_8.png ${TEMP_DIR}/ffd_register_op_8.txt ${BASELINE}/ffd_register_op_8.txt)
ADD_TEST(FFD-Reg-MR-15 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 FALSE              FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_15.png ${TEMP_DIR}/ffd_register_grid_15.png ${TEMP_DIR}/ffd_register_after_diff_15.png ${TEMP_DIR}/ffd_register_before_diff_15.png ${TEMP_DIR}/ffd_register_op_15.txt ${BASELINE}/ffd_register_op_15.txt)
ADD_TEST(FFD-Reg-MR-16 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 TRUE               TRUE               FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_16.png ${TEMP_DIR}/ffd_register_grid_16.png ${TEMP_DIR}/ffd_register_after_diff_16.png ${TEMP_DIR}/ffd_register_before_diff_16.png ${TEMP_DIR}/ffd_register_op_16.txt ${BASELINE}/ffd_register_op_16.txt)
ADD_TEST(FFD-Reg-MR-17 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 TRUE               FALSE              FALSE              FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_17.png ${TEMP_DIR}/ffd_register_grid_17.png ${TEMP_DIR}/ffd_register_after_diff_17.png ${TEMP_DIR}/ffd_register_before_diff_17.png ${TEMP_DIR}/ffd_register_op_17.txt ${BASELINE}/ffd_register_op_17.txt)
ADD_TEST(FFD-Reg-MR-18 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 FALSE              FALSE              TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_18.png ${TEMP_DIR}/ffd_register_grid_18.png ${TEMP_DIR}/ffd_register_after_diff_18.png ${TEMP_DIR}/ffd_register_before_diff_18.png ${TEMP_DIR}/ffd_register_op_18.txt ${BASELINE}/ffd_register_op_18.txt)
ADD_TEST(FFD-Reg-MR-19 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 TRUE               TRUE               TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_19.png ${TEMP_DIR}/ffd_register_grid_19.png ${TEMP_DIR}/ffd_register_after_diff_19.png ${TEMP_DIR}/ffd_register_before_diff_19.png ${TEMP_DIR}/ffd_register_op_19.txt ${BASELINE}/ffd_register_op_19.txt)
ADD_TEST(FFD-Reg-MR-20 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}     FFDRegisterTest ${INPUT_DATA}/mr1.png         ${INPUT_DATA}/mr2.png                      15 15 10 1 TRUE               FALSE              TRUE               FALSE         64   0.000001  ${TEMP_DIR}/ffd_register_op_20.png ${TEMP_DIR}/ffd_register_grid_20.png ${TEMP_DIR}/ffd_register_after_diff_20.png ${TEMP_DIR}/ffd_register_before_diff_20.png ${TEMP_DIR}/ffd_register_op_20.txt ${BASELINE}/ffd_register_op_20.txt)
ADD_TEST(NondirectionalDerivativeOperatorTest ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} NondirectionalDerivativeOperatorTest)
ADD_TEST(SSD-Registration-Force-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} SSDRegistrationForceFilterTest ${INPUT_DATA}/fluid_fixed_10_x_10.png ${INPUT_DATA}/fluid_fixed_10_x_10_1_pixel_diff.png 1.9659821730068243 0.0001 4 2 2287.5 4575 )
ADD_TEST(CC-Registration-Force-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} CrossCorrelationDerivativeForceFilterTest ${INPUT_DATA}/ellipse-128-128-128-50-45-40.nii ${INPUT_DATA}/ellipse-128-128-128-50-50-50.nii 1e-10 76 64 28 2.757567e-06 0 -2.757567e-06)
ADD_TEST(JacTest-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS}  ForwardDifferenceDisplacementFieldJacobianDeterminantFilterTest)

#################################################################################
# Build instructions.
#################################################################################
SET( RegistrationToolboxUnitTests_SRCS
SwitchableAffineTransformTest.cxx
SwitchableAffine2DTransformTest.cxx
EulerAffine2DJacobianTest.cxx
EulerAffine2DTransformTest.cxx
EulerAffine3DTransformTest.cxx
EulerAffine3DJacobianTest.cxx
ImageMetricTest2D.cxx
SingleRes2DMeanSquaresTest.cxx
SingleRes2DCorrelationMaskTest.cxx
SingleRes2DMultiStageMethodTest.cxx
MultiRes2DMeanSquaresTest.cxx
ImageRegistrationFilterTest.cxx
SquaredUCLSimplexTest.cxx
SquaredUCLRegularStepOptimizerTest.cxx
SquaredUCLGradientDescentOptimizerTest.cxx
BSplineTransformTest.cxx
NMILocalHistogramDerivativeForceFilterTest.cxx
itkHistogramRegistrationForceGeneratorTest.cxx
BSplineSmoothTest.cxx
BSplineInterpolateTest.cxx
FFDRegisterTest.cxx
NondirectionalDerivativeOperatorTest.cxx
HistogramParzenWindowDerivativeForceFilterTest.cxx
SingleRes2DBlockMatchingTest.cxx
MatrixLinearCombinationFunctionsTests.cxx
SSDRegistrationForceFilterTest.cxx
CrossCorrelationDerivativeForceFilterTest.cxx
ForwardDifferenceDisplacementFieldJacobianDeterminantFilterTest.cxx
)

ADD_EXECUTABLE(RegistrationToolboxUnitTests RegistrationToolboxUnitTests.cxx ${RegistrationToolboxUnitTests_SRCS})
TARGET_LINK_LIBRARIES(RegistrationToolboxUnitTests ${NIFTK_TEST_EXT_ITK_REGISTRATION_TOOLBOX_LINK_LIBRARIES} )

