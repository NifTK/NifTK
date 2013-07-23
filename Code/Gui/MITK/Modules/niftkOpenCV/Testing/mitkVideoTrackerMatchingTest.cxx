/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoTrackerMatching.h"
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>


//-----------------------------------------------------------------------------
int mitkVideoTrackerMatchingTest(int argc, char** argv)
{

  mitk::VideoTrackerMatching::Pointer Matcher = mitk::VideoTrackerMatching::New();

  return EXIT_SUCCESS;
}
