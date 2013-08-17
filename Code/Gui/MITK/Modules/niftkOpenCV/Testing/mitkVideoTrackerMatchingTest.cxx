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

bool CheckTrackerMatrix (cv::Mat matrix)
{
  double Error = 0.0;

  Error += fabs ( matrix.at<float >(0,0) - ( -0.1499633640 ));
  Error += fabs ( matrix.at<float>(0,1) - ( 0.9741477966 ));
  Error += fabs ( matrix.at<float>(0,2) - ( 0.1689591110 ));
  Error += fabs ( matrix.at<float>(0,3) - ( -562.9057617188 ));

  Error += fabs ( matrix.at<float>(1,0) - ( 0.9324881434 ));
  Error += fabs ( matrix.at<float>(1,1) - ( 0.0825609267 ));
  Error += fabs ( matrix.at<float>(1,2) - ( 0.3516384661 ));
  Error += fabs ( matrix.at<float>(1,3) - ( -68.4594039917 ));
  
  Error += fabs ( matrix.at<float>(2,0) - ( 0.3285983801 ));
  Error += fabs ( matrix.at<float>(2,1) - ( 0.2102852464 ));
  Error += fabs ( matrix.at<float>(2,2) - ( -0.9207623005 ));
  Error += fabs ( matrix.at<float>(2,3) - ( -1978.4907226562 ));
  
  Error += fabs ( matrix.at<float>(3,0) - ( 0.0 ));
  Error += fabs ( matrix.at<float>(3,1) - ( 0.0 ));
  Error += fabs ( matrix.at<float>(3,2) - ( 0.0 ));
  Error += fabs ( matrix.at<float>(3,3) - ( 1.0 ));
  if ( Error < 1e-6 ) 
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------------
int mitkVideoTrackerMatchingTest(int argc, char * argv[])
{
  mitk::VideoTrackerMatching::Pointer Matcher = mitk::VideoTrackerMatching::New();
  Matcher->Initialise(argv[1]);

  //check it initialised, check it gets the right matrix with the right time error
  MITK_TEST_CONDITION_REQUIRED (Matcher->IsReady() , "Testing that VideoTrackerMatcherInitialised OK"); 
  
  long long *TimingError = new long long;
  cv::Mat TrackingMatrix = Matcher->GetTrackerMatrix(20, TimingError, 0);

  //1374066239681720400-1374066239683720400 = -2000000
  MITK_TEST_CONDITION_REQUIRED(*TimingError == -2000000, "Testing Timing error");
  MITK_TEST_CONDITION_REQUIRED(CheckTrackerMatrix(TrackingMatrix), "Testing Tracker Matrix");
  return EXIT_SUCCESS;
}
