/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPCLTest.h"
#include <cstdlib>

int main (int argc, char** argv)
{
  mitk::PCLTest::Pointer test = mitk::PCLTest::New();
  test->Update(argv[1]);
  return EXIT_SUCCESS;
}
