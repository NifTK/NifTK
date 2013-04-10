/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif


// this file defines the itkIOTests for the test driver
// and all it expects is that you have a function called RegisterTests
#include <iostream>
#include "itkTestMain.h" 

void RegisterTests()
{
  // Transforms
  REGISTER_TEST(SwitchableAffine2DTransformTest);
  REGISTER_TEST(SwitchableAffineTransformTest);
  REGISTER_TEST(EulerAffine2DTransformTest);
  REGISTER_TEST(EulerAffine2DJacobianTest);
  REGISTER_TEST(EulerAffine3DTransformTest);
  REGISTER_TEST(EulerAffine3DJacobianTest);
  REGISTER_TEST(BSplineTransformTest);
  
  // Metrics
  REGISTER_TEST(ImageMetricTest2D);
  REGISTER_TEST(MatrixLinearCombinationFunctionsTests); 

  // Optimizers
  REGISTER_TEST(SquaredUCLSimplexTest);
  REGISTER_TEST(SquaredUCLRegularStepOptimizerTest);
  REGISTER_TEST(SquaredUCLGradientDescentOptimizerTest);

  // Block Matching
  REGISTER_TEST(SingleRes2DBlockMatchingTest);
  
  // Deformable stuff.
  REGISTER_TEST(NMILocalHistogramDerivativeForceFilterTest);
  REGISTER_TEST(itkHistogramRegistrationForceGeneratorTest);
  REGISTER_TEST(BSplineSmoothTest);
  REGISTER_TEST(BSplineInterpolateTest);
  REGISTER_TEST(FFDRegisterTest);
  REGISTER_TEST(HistogramParzenWindowDerivativeForceFilterTest);
  REGISTER_TEST(SSDRegistrationForceFilterTest);
  REGISTER_TEST(CrossCorrelationDerivativeForceFilterTest);
  REGISTER_TEST(ForwardDifferenceDisplacementFieldJacobianDeterminantFilterTest); 
  
  // All the rest.
  REGISTER_TEST(ImageRegistrationFilterTest);
  REGISTER_TEST(SingleRes2DMeanSquaresTest);
  REGISTER_TEST(SingleRes2DCorrelationMaskTest);
  REGISTER_TEST(SingleRes2DMultiStageMethodTest);
  REGISTER_TEST(MultiRes2DMeanSquaresTest);
  REGISTER_TEST(NondirectionalDerivativeOperatorTest);
  
}
