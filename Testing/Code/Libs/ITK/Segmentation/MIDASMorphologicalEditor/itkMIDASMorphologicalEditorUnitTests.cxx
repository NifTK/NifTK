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
  REGISTER_TEST(itkMIDASLimitByRegionFunctionTest);
  REGISTER_TEST(itkMIDASMeanIntensityWithinARegionFilterTest);
  REGISTER_TEST(itkMIDASConditionalErosionFilterTest);
  REGISTER_TEST(itkMIDASConditionalDilationFilterTest);
  REGISTER_TEST(itkMIDASDownSamplingFilterTest);
  REGISTER_TEST(itkMIDASUpSamplingFilterTest);
  REGISTER_TEST(itkMIDASRethresholdingFilterTest);
  REGISTER_TEST(itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest);
  REGISTER_TEST(itkMIDASMaskByRegionFilterTest);
  REGISTER_TEST(itkMIDASPipelineTest);
}
