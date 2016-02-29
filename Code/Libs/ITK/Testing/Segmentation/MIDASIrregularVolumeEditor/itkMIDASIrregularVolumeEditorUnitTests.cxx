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
  REGISTER_TEST(itkMIDASImageUpdateCopyRegionProcessorTest);
  REGISTER_TEST(itkMIDASImageUpdateClearRegionProcessorTest);
  REGISTER_TEST(itkMIDASImageUpdatePixelWiseSingleValueProcessorTest);
  REGISTER_TEST(itkMIDASRegionGrowingImageFilterTest2);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorBySlicesTest);
  REGISTER_TEST(itkMIDASRetainMarksNoThresholdingTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorMinimumRegionTest);
}
