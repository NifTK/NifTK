/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkFixedLengthFileReader.h>
#include <cstdlib>
#include <iostream>

/**
 * \brief Basic test harness for FileHelper.h
 */
int niftkFixedLengthFileReaderTest(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage   :niftkFixedLengthFileReaderTest file" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

