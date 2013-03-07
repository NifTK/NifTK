/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTestCornerExtraction.h"

int main(int argc, char** argv)
{
  mitk::TestCornerExtraction::Pointer testObject = mitk::TestCornerExtraction::New();
  std::string fileNameLeft;
  std::string fileNameRight;
  if (argc >= 3)
  {
    fileNameLeft = std::string(argv[1]);
    fileNameRight = std::string(argv[2]);
    testObject->Run(fileNameLeft, fileNameRight);
  }
  else
  {
    std::cerr << "Error: Usage= niftkTestCornerExtraction imageLeft.png imageRight.png" << std::endl;
  }
}
