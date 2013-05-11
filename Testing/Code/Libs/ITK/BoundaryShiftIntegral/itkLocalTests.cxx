/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// this file defines the itkBasicFiltersTest for the test driver
// and all it expects is that you have a function called RegisterTests

#include <itkTestMain.h>

void RegisterTests()
{
  REGISTER_TEST(itkBinaryIntersectWithPaddingImageFilterTest);
  REGISTER_TEST(itkBinaryUnionWithPaddingImageFilterTest);
  REGISTER_TEST(itkMultipleDilateErodeImageFilterTest);
  REGISTER_TEST(itkIntensityNormalisationCalculatorTest);
  REGISTER_TEST(itkSimpleKMeansClusteringImageFilterTest);
  REGISTER_TEST(itkBoundaryShiftIntegralTest);
}

//image1 mean = 3.649667516899693e+02
//image2 mean = 6.943582577570643e+02






