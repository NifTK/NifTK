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

bool CheckTransformedPointVector (std::vector <mitk::WorldPointsWithTimingError> points)
{
  double Error = 0.0;
  //here are some points calculated indepenently
  std::vector <cv::Point3d> frame0000points;
  frame0000points.push_back(cv::Point3d(-11.9645 ,  -25.0246 ,  137.6881));
  frame0000points.push_back(cv::Point3d(-14.1034  ,   1.5127 ,  132.9415 ));
  frame0000points.push_back(cv::Point3d(24.3209  ,   3.3657 ,  126.3173));
  frame0000points.push_back(cv::Point3d(26.4292 ,  -23.5841 ,  131.1581));
  std::vector <cv::Point3d> frame1155points;
  
  frame1155points.push_back(cv::Point3d(-17.28927 ,  -26.38782 ,  128.45949));
  frame1155points.push_back(cv::Point3d( -18.88983  ,   0.32739 ,  124.57826));
  frame1155points.push_back(cv::Point3d(20.06107  ,   2.37748 ,  123.02983  ));
  frame1155points.push_back(cv::Point3d( 21.62069  , -24.75377 ,  126.98338));
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    Error += fabs ( points[0].m_Points[i].m_Point.x - frame0000points[i].x);
    Error += fabs ( points[0].m_Points[i].m_Point.y - frame0000points[i].y);
    Error += fabs ( points[0].m_Points[i].m_Point.z - frame0000points[i].z);

    Error += fabs ( points[1155].m_Points[i].m_Point.x - frame1155points[i].x);
    Error += fabs ( points[1155].m_Points[i].m_Point.y - frame1155points[i].y);
    Error += fabs ( points[1155].m_Points[i].m_Point.z - frame1155points[i].z);
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

bool CheckProjectionErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector) 
{

  double Error = 0.0;
  //these points were not calculated independently, so this is more a regression test
  std::vector <cv::Point2d> leftErrors;
  leftErrors.push_back(cv::Point2d(0.0,0.0));
  leftErrors.push_back(cv::Point2d(39.9234, 0.0393168 ));
  leftErrors.push_back(cv::Point2d(-0.259703, 29.6481));
  leftErrors.push_back(cv::Point2d(19.0695, 11.079));
  
  std::vector <cv::Point2d> rightErrors;
  rightErrors.push_back(cv::Point2d(0.195419, 0.23519));
  rightErrors.push_back(cv::Point2d(40.7176, -0.0442551 ));
  rightErrors.push_back(cv::Point2d(-0.522525, 30.5887 ));
  rightErrors.push_back(cv::Point2d(19.3863, 11.421));
  
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    Error += fabs ( Projector->GetLeftProjectionErrors()[i].x - leftErrors[i].x);
    Error += fabs ( Projector->GetLeftProjectionErrors()[i].y - leftErrors[i].y);

    Error += fabs ( Projector->GetRightProjectionErrors()[i].x - rightErrors[i].x);
    Error += fabs ( Projector->GetRightProjectionErrors()[i].y - rightErrors[i].y);
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
bool CheckReProjectionErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector) 
{

  double Error = 0.0;
  //some errors calculated independently.
  std::vector <cv::Point3d> leftErrors;
  leftErrors.push_back(cv::Point3d(0.0,0.0,0.0));
  leftErrors.push_back(cv::Point3d(2.5, 0.0, 0.0 ));
  leftErrors.push_back(cv::Point3d(0.0, 3.6, 0.0));
  leftErrors.push_back(cv::Point3d(1.2, 1.4, 0.0));
  
  std::vector <cv::Point3d> rightErrors;
  rightErrors.push_back(cv::Point3d(0.0, 0.0, 0.0));
  rightErrors.push_back(cv::Point3d(2.49889, 0.00106, -0.0));;
  rightErrors.push_back(cv::Point3d(-0.00019,3.59972 , 0.0));
  rightErrors.push_back(cv::Point3d(1.19939, 1.40040 , -0.0));
  
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].x - leftErrors[i].x);
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].y - leftErrors[i].y);
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].z - leftErrors[i].z);

    Error += fabs ( Projector->GetRightReProjectionErrors()[i].x - rightErrors[i].x);
    Error += fabs ( Projector->GetRightReProjectionErrors()[i].y - rightErrors[i].y);
    Error += fabs ( Projector->GetRightReProjectionErrors()[i].z - rightErrors[i].z);
  }

  if ( Error < 3e-2 ) 
  {
    return true;
  }
  else
  {
    return false;
  }
}
bool CheckTriangulationErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector) 
{

  double Error = 0.0;
  //some errors calculated independently.
  std::vector <cv::Point3d> leftErrors;
  leftErrors.push_back(cv::Point3d(0.0,0.0,0.0));
  leftErrors.push_back(cv::Point3d(-2.5, 0.0, 0.0 ));
  leftErrors.push_back(cv::Point3d(0.0, -3.6, 0.0));
  leftErrors.push_back(cv::Point3d(-1.2, -1.4, 0.0));
  
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    Error += fabs ( Projector->GetTriangulationErrors()[i].x - leftErrors[i].x);
    Error += fabs ( Projector->GetTriangulationErrors()[i].y - leftErrors[i].y);
    Error += fabs ( Projector->GetTriangulationErrors()[i].z - leftErrors[i].z);
  }

  if ( Error < 2e-2 ) 
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
  MITK_TEST_BEGIN("mitkProjectPointsOnStereoVideoTest");
  mitk::ProjectPointsOnStereoVideo::Pointer Projector = mitk::ProjectPointsOnStereoVideo::New();
  Projector->SetVisualise(false);
  Projector->Initialise(argv[1], argv[2]);
  Projector->SetTrackerIndex(0);
  mitk::VideoTrackerMatching::Pointer matcher = mitk::VideoTrackerMatching::New();
  matcher->Initialise(argv[1]);
  matcher->SetFlipMatrices(false);
  Projector->SetMatcherCameraToTracker(matcher);
  //check it initialised, check it gets the right matrix with the right time error
  MITK_TEST_CONDITION_REQUIRED (Projector->GetInitOK() , "Testing mitkProjectPointsOnStereoVideo Initialised OK"); 

  //here are the on screen points manually found in frames 0 and 1155
  std::vector < mitk::ProjectedPointPair > frame0000ScreenPoints;
  std::vector < mitk::ProjectedPointPair > frame1155ScreenPoints;
  std::vector < mitk::ProjectedPointPair > frame1400ScreenPoints;
  std::vector < unsigned int > frame0000framenumbers;
  std::vector < unsigned int > frame1155framenumbers;
  std::vector < unsigned int > frame1400framenumbers;
  frame0000ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(756,72), cv::Point2d(852,84 )) );
  frame0000ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(1426,78), cv::Point2d(1524,90 )) );
  frame0000ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(1406,328), cv::Point2d(1506,342 )) );
  frame0000ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(702,306), cv::Point2d(798,320 )) );
  for ( unsigned int i = 0 ; i < 4 ; i ++ ) 
  {
    frame0000framenumbers.push_back(2);
  }
 frame1155ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(668,34), cv::Point2d(762,52 )) );
  frame1155ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(1378,50), cv::Point2d(1474,62 )) );
  frame1155ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(1372,308), cv::Point2d(1468,324)) );
  frame1155ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(628,296), cv::Point2d( 714,308)) );
  for ( unsigned int i = 0 ; i < 4 ; i ++ ) 
  {
    frame1155framenumbers.push_back(1155);
  }
  frame1400ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(438,32), cv::Point2d(340,10 )) );
  frame1400ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(1016,162), cv::Point2d(930,142 )) );
  frame1400ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(798,386), cv::Point2d(714,368)) );
  frame1400ScreenPoints.push_back (mitk::ProjectedPointPair 
      ( cv::Point2d(216,240), cv::Point2d( 122,220)) );
  for ( unsigned int i = 0 ; i < 4 ; i ++ ) 
  {
    frame1400framenumbers.push_back(1400);
  }

  std::vector <mitk::WorldPoint > WorldGridPoints;
  //these are the corners of the grid according to the handeye calibration of the certus
  WorldGridPoints.push_back ( mitk::WorldPoint (cv::Point3d(-826.2,-207.2,-2010.6)));
  WorldGridPoints.push_back ( mitk::WorldPoint (cv::Point3d(-820.3,-205.0,-2036.9)));
  WorldGridPoints.push_back ( mitk::WorldPoint (cv::Point3d(-820.8,-166.1,-2033.7)));
  WorldGridPoints.push_back ( mitk::WorldPoint (cv::Point3d(-826.8,-168.4,-2007.0)));

  Projector->SetWorldPoints(WorldGridPoints);
 
//  Projector->SetWorldPointsByTriangulation(frame1155ScreenPoints,1155);
  Projector->SetWorldPointsByTriangulation(frame0000ScreenPoints,frame0000framenumbers, matcher);
  Projector->SetWorldPointsByTriangulation(frame1400ScreenPoints,frame1400framenumbers, matcher);

 Projector->Project(matcher);
  MITK_TEST_CONDITION_REQUIRED (Projector->GetProjectOK(), "Testing mitkProjectPointsOnStereoVideo projected OK"); 

  MITK_TEST_CONDITION(CheckTransformedPointVector(Projector->GetPointsInLeftLensCS()), "Testing projected points");

  //these are the gold standard projected points for frame 1155
  //1155 664.844 69.984 753.793 68.306 628.092 279.283 711.968 279.424 1264.44 296.217 1365.82 296.783 1277.2 79.8817 1380.06 75.8718
  //these are perturbed world points that should yield the following errors for frame 1155
  Projector->ClearWorldPoints();
  WorldGridPoints.clear();
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-826.2000 ,  -207.2000 ,-2010.6000 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-820.4406 ,  -202.5256 , -2036.5725 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-820.5379 ,  -165.6142 , -2037.2575 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-826.7656 ,  -167.0234 , -2008.2263 )));
  Projector->SetWorldPoints(WorldGridPoints);
  Projector->SetClassifierWorldPoints(WorldGridPoints);
  Projector->Project(matcher);

  std::vector < mitk::GoldStandardPoint> leftGS;
  std::vector < mitk::GoldStandardPoint> rightGS;
  leftGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(664.844, 69.984)));
  leftGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(628.092, 279.283)));
  leftGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1264.44, 296.217)));
  leftGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1277.2, 79.8817)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(753.793, 68.306)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(711.968, 279.424)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1365.82, 296.783)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1380.06, 75.8718)));
  Projector->SetLeftGoldStandardPoints(leftGS);
  Projector->SetRightGoldStandardPoints(rightGS);
  Projector->CalculateProjectionErrors("");
  Projector->CalculateTriangulationErrors("", matcher);

  MITK_TEST_CONDITION(CheckProjectionErrors(Projector), "Testing projection Errors");
  MITK_TEST_CONDITION(CheckReProjectionErrors(Projector), "Testing re-projection Errors");
  MITK_TEST_CONDITION(CheckTriangulationErrors(Projector), "Testing triangulation Errors");

  MITK_TEST_END();
}
