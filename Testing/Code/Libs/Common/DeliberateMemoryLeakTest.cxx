/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-22 18:54:07 +0100 (Sun, 22 Aug 2010) $
 Revision          : $Revision: 4140 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include "ConversionUtils.h"
#include <math.h>
#include <iostream>
#include "stdlib.h"

/**
 * Basic test harness that is meant to create a memory leak, so we can make sure valgrind is working.
 * This isn't a unit test as such, as it always returns EXIT_SUCCESS, but we are simply making sure
 * that valgrind is running unit tests, and checking for memory leaks.
 */
int DeliberateMemoryLeakTest(int argc, char * argv[])
{

  int *dummyArray = new int[1000];
  std::cout << "dummyArray=" << dummyArray << std::endl; // to remove compiler warning about unused variable.
  return EXIT_SUCCESS;
}

