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
  Projector->SetVisualise(true);
  Projector->Initialise(argv[1], argv[2]);
  Projector->SetFlipMatrices(true);
  Projector->SetTrackerIndex(2);
  Projector->SetDrawAxes(true);
  //check it initialised, check it gets the right matrix with the right time error
  MITK_TEST_CONDITION_REQUIRED (Projector->GetInitOK() , "Testing mitkProjectPointsOnStereoVideo Initialised OK"); 

  //here are the on screen points manually found in frames 0 and 1155
  std::vector < std::pair < cv::Point2f, cv::Point2f > > frame0000ScreenPoints;
  std::vector < std::pair < cv::Point2f, cv::Point2f > > frame1155ScreenPoints;
  std::vector < std::pair < cv::Point2f, cv::Point2f > > frame1400ScreenPoints;
  frame0000ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(756,72), cv::Point(852,84 )) );
  frame0000ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(1426,78), cv::Point(1524,90 )) );
  frame0000ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(1406,328), cv::Point(1506,342 )) );
  frame0000ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(702,306), cv::Point(798,320 )) );
 frame1155ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(668,34), cv::Point(762,52 )) );
  frame1155ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(1378,50), cv::Point(1474,62 )) );
  frame1155ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(1372,308), cv::Point(1468,324)) );
  frame1155ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(628,296), cv::Point( 714,308)) );
 frame1400ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(438,32), cv::Point(340,10 )) );
  frame1400ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(1016,162), cv::Point(930,142 )) );
  frame1400ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(798,386), cv::Point(714,368)) );
  frame1400ScreenPoints.push_back (std::pair<cv::Point2f,cv::Point2f>
      ( cv::Point2f(216,240), cv::Point( 122,220)) );

  Projector->SetWorldPointsByTriangulation(frame1400ScreenPoints,1400);
//  Projector->SetWorldPointsByTriangulation(frame1155ScreenPoints,1155);
  Projector->SetWorldPointsByTriangulation(frame0000ScreenPoints,2);
 
  Projector->SetDrawLines(true);

  std::vector <cv::Point3f> WorldGridPoints;
  //these are the corners of the grid according to the handeye calibration of the certus
  WorldGridPoints.push_back ( cv::Point3f(-765.6784 ,-99.4104,1851.3566));
  WorldGridPoints.push_back ( cv::Point3f(-781.6574 ,-62.9164,1863.0450));
  WorldGridPoints.push_back ( cv::Point3f(-754.2505 ,-50.4048,1862.8514));
  WorldGridPoints.push_back ( cv::Point3f(-738.2715 ,-86.8989,1851.1630));
 // Projector->SetWorldPoints(WorldGridPoints);
  Projector->Project();
  MITK_TEST_CONDITION_REQUIRED (Projector->GetProjectOK(), "Testing mitkProjectPointsOnStereoVideo projected OK"); 

  MITK_TEST_CONDITION(CheckTransformedPointVector(Projector->GetPointsInLeftLensCS()), "Testing projected points");
  return EXIT_SUCCESS;
}
