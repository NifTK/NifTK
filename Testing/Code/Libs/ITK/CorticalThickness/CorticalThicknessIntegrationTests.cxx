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

#include <iostream>
#include "itkTestMain.h" 

void RegisterTests()
{
  REGISTER_TEST(LaplacianSolverImageFilterTest);
  REGISTER_TEST(ScalarImageToNormalizedGradientVectorImageFilterTest);
  REGISTER_TEST(StreamlinesFilterTest);
  REGISTER_TEST(CorrectGMUsingPVMapTest);
  REGISTER_TEST(CorrectGMUsingNeighbourhoodTest);
  REGISTER_TEST(LagrangianInitializedStreamlinesFilterTest);
  REGISTER_TEST(Bourgeat2008Test);
  REGISTER_TEST(FourthOrderRungeKuttaVelocityFieldIntegrationTest);
  REGISTER_TEST(FourthOrderRungeKuttaVelocityFieldThicknessTest);
  REGISTER_TEST(DemonsRegistrationFilterTest);
  REGISTER_TEST(DemonsRegistrationFilterUpdateTest);
  REGISTER_TEST(AddUpdateToTimeVaryingVelocityFilterTest);
  REGISTER_TEST(GaussianSmoothVectorFieldFilterTest);
  REGISTER_TEST(RegistrationBasedCorticalThicknessFilterTest);
}
