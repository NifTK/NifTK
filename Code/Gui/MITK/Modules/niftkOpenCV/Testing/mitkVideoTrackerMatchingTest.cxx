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

bool CheckTrackerMatrix (cv::Mat)
{
  double Error = 0.0;
}


//-----------------------------------------------------------------------------
int mitkVideoTrackerMatchingTest(int argc, char** argv)
{

  mitk::VideoTrackerMatching::Pointer Matcher = mitk::VideoTrackerMatching::New();
  Matcher->Initialise(argv[1]);

  //check it initialised, check it gets the right matrix with the right time error
  MITK_TEST_CONDITION_REQUIRED (Matcher->IsReady() , "Testing that VideoTrackerMatcherInitialised OK"); 
  
  long *TimingError = new long;
  cv::Mat TrackingMatrix = Matcher->GetTrackerMatrix(20, TimingError, 0);

  //1374066239681720400-1374066239683720400 = -2000000
  MITK_TEST_CONDITION_REQUIRED(*TimingError == -2000000, "Testing Timing error");
  return EXIT_SUCCESS;
}
