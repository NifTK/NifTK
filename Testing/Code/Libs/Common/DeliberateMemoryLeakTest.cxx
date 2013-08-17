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
#include <ConversionUtils.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>

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

