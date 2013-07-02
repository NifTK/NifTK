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
#include <itkTestMain.h>

void RegisterTests()
{
  REGISTER_TEST(BoundaryValueRescaleIntensityImageFilterTest);
  REGISTER_TEST(SetOutputVectorToCurrentPositionFilterTest);
  REGISTER_TEST(VectorMagnitudeImageFilterTest);
  REGISTER_TEST(VectorVPlusLambdaUImageFilterTest);
  REGISTER_TEST(ShapeBasedAveragingImageFilterTest); 
  REGISTER_TEST(ExtractEdgeImageTest);
  REGISTER_TEST(MeanCurvatureImageFilterTest);
  REGISTER_TEST(GaussianCurvatureImageFilterTest);
  REGISTER_TEST(itkExcludeImageFilterTest);
  REGISTER_TEST(itkLargestConnectedComponentFilterTest);
}
