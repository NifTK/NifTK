/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVTest.h"

int main(int argc, char** argv)
{
  mitk::OpenCVTest::Pointer testObject = mitk::OpenCVTest::New();
  std::string fileName;
  if (argc >= 2)
  {
    fileName = std::string(argv[1]);
  }
  testObject->Run(fileName);
}
