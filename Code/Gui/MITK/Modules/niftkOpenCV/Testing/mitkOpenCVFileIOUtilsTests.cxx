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

#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkOpenCVFileIOUtils.h>
#include <cmath>

/**
 * \file Tests for some of the functions in openCVFileIOUtils.
 */

void LoadTimeStampedPointsTest(std::string dir)
{
  std::string pointdir = dir + "points";
  std::string matrixdir = dir + "matrices";

  std::vector < std::pair<unsigned long long, cv::Point3d> > timeStampedPoints = mitk::LoadTimeStampedPoints(pointdir);
  
  MITK_TEST_CONDITION ( timeStampedPoints.size() == 88 , "Testing 88 points were loaded. " << timeStampedPoints.size() );
  MITK_TEST_CONDITION ( timeStampedPoints[0].first == 1389084822590698200 , "Testing first time stamp " <<  timeStampedPoints[0].first);
  MITK_TEST_CONDITION ( timeStampedPoints[87].first == 1389085072647000600 , "Testing last time stamp " <<  timeStampedPoints[87].first);
  
//  MITK_TEST_CONDITION ( mitk::NearlyEqual (timeStampedPoints[0].second == 1389084822590698200 , "Testing first time stamp " <<  timeStampedPoints[0].first);
 // MITK_TEST_CONDITION ( timeStampedPoints[87].first == 1389085072647000600 , "Testing last time stamp " <<  timeStampedPoints[87].first);

}


int mitkOpenCVFileIOUtilsTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVFileIOUtilsTests");

  LoadTimeStampedPointsTest(argv[1]);
  MITK_TEST_END();
}



