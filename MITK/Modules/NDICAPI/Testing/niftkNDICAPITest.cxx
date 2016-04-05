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
#include <niftkNDICAPITracker.h>
#include <mitkTestingMacros.h>
#include <iostream>

/**
* \file niftkNDICAPITest.cxx
* \brief Tests connection to NDI tracker (only works if one exists). 
*/
int niftkNDICAPITest(int argc, char * argv[])
{
  // Always start with this, with name of function.
  MITK_TEST_BEGIN("niftkNDICAPITest");

  niftk::NDICAPITracker tracker;

  if (tracker.Probe() != niftk::NDICAPITracker::PLUS_SUCCESS)
  {
    std::cerr << "Failed to probe" << std::endl;
    return EXIT_FAILURE;
  }

  // Always end with this.
  MITK_TEST_END();
}

