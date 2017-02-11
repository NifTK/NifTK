/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkUSReconstructor.h>
#include <mitkTestingMacros.h>

/**
 * \file Test harness for niftk::USReconstructor.
 */
int niftkUSReconTest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkUSReconTest");

  if (argc != 1)
  {
    MITK_ERROR << "Usage: niftkUSReconTest describe arguments here.";
    return EXIT_FAILURE;
  }

  niftk::USReconstructor::Pointer reconstructor = niftk::USReconstructor::New();

  MITK_TEST_CONDITION_REQUIRED(reconstructor.IsNotNull(), "Checking reconstructor was constructed.");
  MITK_TEST_CONDITION_REQUIRED(1 == 2,"... Implement tests here.");
  MITK_TEST_END();
}
