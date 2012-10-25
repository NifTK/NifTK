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
