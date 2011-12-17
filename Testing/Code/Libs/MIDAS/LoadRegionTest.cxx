/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3721 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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
