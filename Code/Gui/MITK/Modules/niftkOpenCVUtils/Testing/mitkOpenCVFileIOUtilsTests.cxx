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
#include <mitkOpenCVMaths.h>
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
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual (timeStampedPoints[0].second,cv::Point3d (315.0,397.0,0.0),1e-6), "Testing first point value " <<  timeStampedPoints[0].second);
  MITK_TEST_CONDITION ( mitk::NearlyEqual (timeStampedPoints[87].second, cv::Point3d (338.0, 359.0, 0.0),1e-6), "Testing last time stamp " <<  timeStampedPoints[87].second);

}


int mitkOpenCVFileIOUtilsTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVFileIOUtilsTests");

  LoadTimeStampedPointsTest(argv[1]);
  MITK_TEST_END();
}



