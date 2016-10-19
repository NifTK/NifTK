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
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkIOUtil.h>
#include <niftkBinaryMaskUtils.h>
#include <niftkImageUtils.h>

/**
 * Basic test harness for niftkBinaryMaskUtils.
 */
int niftkBinaryMaskTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkBinaryMaskTest");

  if (argc != 5)
  {
    mitkThrow() << "Usage: niftkBinaryMaskUtils input1.png input2.png mode[int] expectedOutput.png";
  }

  mitk::Image::Pointer im1 = mitk::IOUtil::LoadImage(argv[1]);
  mitk::Image::Pointer im2 = mitk::IOUtil::LoadImage(argv[2]);
  int mode = atoi(argv[3]);
  mitk::Image::Pointer expected = mitk::IOUtil::LoadImage(argv[4]);
  mitk::Image::Pointer output = im2->Clone();

  if (mode == 0)
  {
    niftk::BinaryMaskAndOperator(im1, im2, output);
  }
  else if (mode == 1)
  {
    niftk::BinaryMaskOrOperator(im1, im2, output);
  }
  else
  {
    mitkThrow() << "Unexpected mode:" << mode;
  }

  MITK_TEST_CONDITION_REQUIRED(niftk::ImagesHaveEqualIntensities(output, expected), "... Checking output and expected have equal intensities.");

  MITK_TEST_END();
}

