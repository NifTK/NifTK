/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-17 11:52:07 +0100 (Mon, 17 Oct 2011) $
 Revision          : $Revision: 7531 $
 Last modified by  : $Author: sj $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include "itkTestMain.h"

void RegisterTests()
{
  REGISTER_TEST(itkMIDASMeanIntensityWithinARegionFilterTest);
  REGISTER_TEST(itkMIDASThresholdAndAxialCutoffFilterTest);
  REGISTER_TEST(itkMIDASConditionalErosionFilterTest);
  REGISTER_TEST(itkMIDASConditionalDilationFilterTest);
  REGISTER_TEST(itkMIDASLargestConnectedComponentFilterTest);
  REGISTER_TEST(itkMIDASDownSamplingFilterTest);
  REGISTER_TEST(itkMIDASUpSamplingFilterTest);
  REGISTER_TEST(itkMIDASRethresholdingFilterTest);
  REGISTER_TEST(itkMIDASRegionGrowingImageFilterTest2);
  REGISTER_TEST(itkExcludeImageFilterTest);
  REGISTER_TEST(itkLimitByRegionFunctionTest);
  REGISTER_TEST(itkImageUpdateCopyRegionProcessorTest);
  REGISTER_TEST(itkImageUpdateClearRegionProcessorTest);
  REGISTER_TEST(itkImageUpdatePixelWiseSingleValueProcessorTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorTest);
  REGISTER_TEST(itkMIDASWipeSliceTest);
  REGISTER_TEST(itkMIDASWipePlusTest);
  REGISTER_TEST(itkMIDASWipeMinusTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorBySlicesTest);
  REGISTER_TEST(itkMIDASRetainMarksNoThresholdingTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorSplitExistingRegionTest);
  REGISTER_TEST(itkMIDASRegionGrowingProcessorTest);
  REGISTER_TEST(itkMIDASPropagateUpProcessorTest);
  REGISTER_TEST(itkMIDASPropagateDownProcessorTest);
  REGISTER_TEST(itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest);
  REGISTER_TEST(itkMIDASRegionOfInterestCalculatorMinimumRegionTest);
  REGISTER_TEST(itkMIDASThresholdApplyProcessorTest);
}
