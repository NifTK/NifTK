/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectPointsOnStereoVideo.h"
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

bool CheckTransformedPointVector (std::vector < std::vector <cv::Point3f> > points)
{
  double Error = 0.0;
  //here are some points calculated indepenently
  std::vector <cv::Point3f> frame0000points;
  frame0000points.push_back(cv::Point3f(35.8970  ,  36.3999 ,  124.6265));
  frame0000points.push_back(cv::Point3f(76.4005  ,  40.0814 ,  116.2794));
  frame0000points.push_back(cv::Point3f(73.0312 ,   69.5444 ,  110.9602));
  frame0000points.push_back(cv::Point3f(32.5277 ,   65.8629 ,  119.3073));
  std::vector <cv::Point3f> frame1155points;
  frame1155points.push_back(cv::Point3f(41.3955 ,   38.0281 ,  123.3948));
  frame1155points.push_back(cv::Point3f(82.2175 ,   36.2113 ,  116.0442));
  frame1155points.push_back(cv::Point3f(82.9025 ,   65.7807  , 110.3089 ));
  frame1155points.push_back(cv::Point3f(42.0805  ,  67.5975 ,  117.6595));
  
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    Error += fabs ( points[0][i].x - frame0000points[i].x);
    Error += fabs ( points[0][i].y - frame0000points[i].y);
    Error += fabs ( points[0][i].z - frame0000points[i].z);
    //MITK_INFO << Error;

    Error += fabs ( points[1155][i].x - frame1155points[i].x);
    Error += fabs ( points[1155][i].y - frame1155points[i].y);
    Error += fabs ( points[1155][i].z - frame1155points[i].z);
    //MITK_INFO << Error;
  }


  if ( Error < 2e-3 ) 
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------------
int mitkProjectPointsOnStereoVideoTest(int argc, char * argv[])
{
  mitk::ProjectPointsOnStereoVideo::Pointer Projector = mitk::ProjectPointsOnStereoVideo::New();
  Projector->Initialise(argv[1], argv[2]);
  Projector->SetFlipMatrices(true);
  Projector->SetTrackerIndex(2);

  //check it initialised, check it gets the right matrix with the right time error
  MITK_TEST_CONDITION_REQUIRED (Projector->GetInitOK() , "Testing mitkProjectPointsOnStereoVideo Initialised OK"); 

  std::vector <cv::Point3f> WorldGridPoints;
  //these are the corners of the grid according to the handeye calibration of the certus
  WorldGridPoints.push_back ( cv::Point3f(-765.6784 ,-99.4104,1851.3566));
  WorldGridPoints.push_back ( cv::Point3f(-781.6574 ,-62.9164,1863.0450));
  WorldGridPoints.push_back ( cv::Point3f(-754.2505 ,-50.4048,1862.8514));
  WorldGridPoints.push_back ( cv::Point3f(-738.2715 ,-86.8989,1851.1630));
  Projector->SetWorldPoints(WorldGridPoints);
  Projector->Project();
  MITK_TEST_CONDITION_REQUIRED (Projector->GetProjectOK(), "Testing mitkProjectPointsOnStereoVideo projected OK"); 

  MITK_TEST_CONDITION(CheckTransformedPointVector(Projector->GetPointsInLeftLensCS()), "Testing projected points");
  return EXIT_SUCCESS;
}
