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

#include "stdlib.h"
#include "MidasIO.h"
#include "ConversionUtils.h"
#include <iostream>

int LoadRegionTest(int argc, char * argv[])
{
  if (argc != 4)
  {
    std::cerr << "Usage: LoadRegionTest image region volume" << std::endl;
    return EXIT_FAILURE;
  }
  DummyLoadRegion();

  std::string image = argv[1];
  std::string region = argv[2];
  float expectedVolume = niftk::ConvertToDouble(argv[3]);

  std::cout << "Started with image=" << image << ", region=" << region << ", expectedVolume=" << expectedVolume << std::endl;

  return EXIT_SUCCESS;
}
