/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectPointsOnStereoVideo.h"
#include <niftkFileHelper.h>
#include <mitkOpenCVMaths.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

bool CheckTransformedPointVector (std::vector <mitk::PickedPointList::Pointer> points, unsigned int expectedVectorSize )
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

  if ( points[1155]->GetChannel() != "left_lens" )
  {
    MITK_ERROR << "Points in left lens returned with wrong channel name " << points[1155]->GetChannel();
    return false;
  }

  if ( points[1155]->GetFrameNumber() != 1155 )
  {
    MITK_ERROR << "Points in left lens returned with wrong frame number " << points[1155]->GetFrameNumber();
    return false;
  }

  if ( points[1155]->GetTimeStamp() != 1374854339638394200 ) //work this out, by default the matcher has no lag, so just check framemap for timestamp and find nearest in tracking director
  {
    MITK_ERROR << "Points in left lens returned with wrong time stamp " << points[1155]->GetTimeStamp();
    return false;
  }
  std::vector < mitk::PickedObject > frame0Points = points[0]->GetPickedObjects();
  std::vector < mitk::PickedObject > frame1155Points = points[1155]->GetPickedObjects();

  if ( ( frame0Points.size() != expectedVectorSize ) || frame1155Points.size() != expectedVectorSize )
  {
    MITK_ERROR << "Points in left lens returned with wrong point vector size " << frame0Points.size() << " , " << frame1155Points.size();
    return false;
  }

  for ( int i = 0 ; i < 4 ; i ++ )
  {
    Error += fabs ( frame0Points[i].m_Points[0].x - frame0000points[i].x);
    Error += fabs ( frame0Points[i].m_Points[0].y - frame0000points[i].y);
    Error += fabs ( frame0Points[i].m_Points[0].z - frame0000points[i].z);

    Error += fabs ( frame1155Points[i].m_Points[0].x - frame1155points[i].x);
    Error += fabs ( frame1155Points[i].m_Points[0].y - frame1155points[i].y);
    Error += fabs ( frame1155Points[i].m_Points[0].z - frame1155points[i].z);

    if ( ( frame0Points[i].m_TimeStamp != 1374854320028272600 ) || ( frame1155Points[i].m_TimeStamp != 1374854339638394200 ) )
    {
      MITK_ERROR << "Points in left lens returned with wrong timestamp for set  " << i << " : " << frame0Points[i].m_TimeStamp << " , " << frame1155Points[i].m_TimeStamp;
      return false;
    }

    if ( ( frame0Points[i].m_FrameNumber != 0 ) || ( frame1155Points[i].m_FrameNumber != 1155 ) )
    {
      MITK_ERROR << "Points in left lens returned with wrong frame number for set  " << i << " : " << frame0Points[i].m_FrameNumber << " , " << frame1155Points[i].m_FrameNumber;
      return false;
    }


    if ( ( frame0Points[i].m_Points.size() != 1 ) || ( frame1155Points[i].m_Points.size() != 1 ) )
    {
      MITK_ERROR << "Points in left lens returned with wrong point vector size for set  " << i << " : " << frame0Points[i].m_Points.size() << " , " << frame1155Points[i].m_Points.size();
      return false;
    }

    if ( ( frame0Points[i].m_Channel != "left_lens" ) || ( frame1155Points[i].m_Channel != "left_lens" ) )
    {
      MITK_ERROR << "Points in left lens returned with wrong channel for set  " << i << " : " << frame0Points[i].m_Channel << " , " << frame1155Points[i].m_Channel;
      return false;
    }

    if ( ( frame0Points[i].m_IsLine ) || ( frame1155Points[i].m_IsLine ) )
    {
      MITK_ERROR << "Points in left lens returned with wrong point type set for set  " << i << " : " << frame0Points[i].m_IsLine << " , " << frame1155Points[i].m_IsLine;
      return false;
    }
  }

  if ( Error < 2e-3 )
  {
    return true;
  }
  else
  {
    MITK_ERROR << "Absolute error = " << Error << ": too high for projected point vector, failing.";
    return false;
  }
}

bool CheckProjectionErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector)
{
  if ( Projector->GetLeftProjectionErrors().size() != 4 || Projector->GetRightProjectionErrors().size() != 4 )
  {
    MITK_ERROR << "Wrong number of points in projected error vector";
    return false;
  }

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
    Error += fabs ( Projector->GetLeftProjectionErrors()[i].m_Points[0].x - leftErrors[i].x);
    Error += fabs ( Projector->GetLeftProjectionErrors()[i].m_Points[0].y - leftErrors[i].y);

    Error += fabs ( Projector->GetRightProjectionErrors()[i].m_Points[0].x - rightErrors[i].x);
    Error += fabs ( Projector->GetRightProjectionErrors()[i].m_Points[0].y - rightErrors[i].y);
  }

  if ( Error < 2e-3 )
  {
    return true;
  }
  else
  {
    MITK_ERROR << "Absolute projection error = " << Error << ": too high for projected point vector, failing.";
    return false;
  }
}
bool CheckReProjectionErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector)
{
  if ( Projector->GetLeftReProjectionErrors().size() != 4 || Projector->GetRightReProjectionErrors().size() != 4 )
  {
    MITK_ERROR << "Wrong number of points in reprojected error vector";
    return false;
  }

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
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].m_Points[0].x - leftErrors[i].x);
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].m_Points[0].y - leftErrors[i].y);
    Error += fabs ( Projector->GetLeftReProjectionErrors()[i].m_Points[0].z - leftErrors[i].z);

    Error += fabs ( Projector->GetRightReProjectionErrors()[i].m_Points[0].x - rightErrors[i].x);
    Error += fabs ( Projector->GetRightReProjectionErrors()[i].m_Points[0].y - rightErrors[i].y);
    Error += fabs ( Projector->GetRightReProjectionErrors()[i].m_Points[0].z - rightErrors[i].z);
  }

  if ( Error < 4e-2 )
  {
    return true;
  }
  else
  {
    MITK_ERROR << "Absolute reprojection error = " << Error << ": too high for projected point vector, failing.";
    return false;
  }
}

//-----------------------------------------------------------------------------
bool CheckTriangulationErrors (mitk::ProjectPointsOnStereoVideo::Pointer Projector)
{

  if ( Projector->GetTriangulationErrors().size() != 4 )
  {
    MITK_ERROR << "Wrong number of points in triangulated error vector";
    return false;
  }

  double Error = 0.0;
  //some errors calculated independently.
  std::vector <cv::Point3d> leftErrors;
  leftErrors.push_back(cv::Point3d(0.0,0.0,0.0));
  leftErrors.push_back(cv::Point3d(-2.5, 0.0, 0.0 ));
  leftErrors.push_back(cv::Point3d(0.0, -3.6, 0.0));
  leftErrors.push_back(cv::Point3d(-1.2, -1.4, 0.0));

  for ( int i = 0 ; i < 4 ; i ++ )
  {
    Error += fabs ( Projector->GetTriangulationErrors()[i].m_Points[0].x - leftErrors[i].x);
    Error += fabs ( Projector->GetTriangulationErrors()[i].m_Points[0].y - leftErrors[i].y);
    Error += fabs ( Projector->GetTriangulationErrors()[i].m_Points[0].z - leftErrors[i].z);
  }

  if ( Error < 2e-2 )
  {
    return true;
  }
  else
  {
    MITK_ERROR << "Absolute triangulation error = " << Error << ": too high for projected point vector, failing.";
    return false;
  }
}

//-----------------------------------------------------------------------------
bool CheckTriangulateGoldStandardPoints (mitk::ProjectPointsOnStereoVideo::Pointer Projector, mitk::VideoTrackerMatching::Pointer matcher, const std::string& outputDir)
{

  std::string tempFile = outputDir + niftk::GetFileSeparator() + "triangulationTest.mps";
  Projector->SetTriangulatedPointsOutName(tempFile);
  Projector->TriangulateGoldStandardPoints (matcher);

  mitk::PointSet::Pointer pointSet = mitk::IOUtil::LoadPointSet ( tempFile );

  //check there are four points
  if ( pointSet->GetSize() != 4 )
  {
    MITK_ERROR << "Wrong number of triangulated points, " << pointSet->GetSize() << " ne 4.";
    return false;
  }

  std::vector < cv::Point3d > triangulatedPointsVector = mitk::PointSetToVector ( pointSet );

  double error = 0;

  error += fabs ( triangulatedPointsVector[0].x - ( -826.2 ) ) ;
  error += fabs ( triangulatedPointsVector[0].y - ( -207.2 ) ) ;
  error += fabs ( triangulatedPointsVector[0].z - ( -2010.6 ) ) ;

  error += fabs ( triangulatedPointsVector[1].x - (-820.3  ) ) ;
  error += fabs ( triangulatedPointsVector[1].y - ( -205.0 ) ) ;
  error += fabs ( triangulatedPointsVector[1].z - (-2036.9  ) ) ;

  error += fabs ( triangulatedPointsVector[2].x - ( -820.8 ) ) ;
  error += fabs ( triangulatedPointsVector[2].y - ( -166.1 ) ) ;
  error += fabs ( triangulatedPointsVector[2].z - ( -2033.7 ) ) ;

  error += fabs ( triangulatedPointsVector[3].x - ( -826.8 ) ) ;
  error += fabs ( triangulatedPointsVector[3].y - ( -168.4 ) ) ;
  error += fabs ( triangulatedPointsVector[3].z - ( -2007.0 ) ) ;

  if ( error < 2e-2 )
  {
    return true;
  }
  else
  {
    MITK_ERROR << "Absolute error for TriangulateGoldStandardPoints = " << error << ": too high for projected point vector, failing.";
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

  Projector->AppendWorldPoints(WorldGridPoints);

//  Projector->SetWorldPointsByTriangulation(frame1155ScreenPoints,1155);
  Projector->AppendWorldPointsByTriangulation(frame0000ScreenPoints,frame0000framenumbers, matcher);
  Projector->AppendWorldPointsByTriangulation(frame1400ScreenPoints,frame1400framenumbers, matcher);

  Projector->Project(matcher);
  MITK_TEST_CONDITION_REQUIRED (Projector->GetProjectOK(), "Testing mitkProjectPointsOnStereoVideo projected OK");

  MITK_TEST_CONDITION(CheckTransformedPointVector(Projector->GetPointsInLeftLensCS(), 12), "Testing projected points");

  Projector->ClearWorldPoints();

  mitk::PickedPointList::Pointer ModelGridPoints = mitk::PickedPointList::New();

  ModelGridPoints->SetInOrderedMode(true);
  ModelGridPoints->AddPoint ( cv::Point3d (-1.416343, -0.452355, 0.051662), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (-0.612092, 26.578511, -0.110197), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (38.401987, 25.359090, -0.449377), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (37.519204, -2.088515, -0.274778), cv::Scalar ( 255,255,255));

  cv::Mat* modelToWorldTransform = new cv::Mat(4,4,CV_64FC1);
  modelToWorldTransform->at<double>(0,0) = -0.0146565;
  modelToWorldTransform->at<double>(0,1) = 0.212855;
  modelToWorldTransform->at<double>(0,2) = -0.976974;
  modelToWorldTransform->at<double>(0,3) = -826.074;

  modelToWorldTransform->at<double>(1,0) = 0.998658;
  modelToWorldTransform->at<double>(1,1) = 0.051653;
  modelToWorldTransform->at<double>(1,2) = -0.00372809;
  modelToWorldTransform->at<double>(1,3) = -205.762;

  modelToWorldTransform->at<double>(2,0) = 0.049670;
  modelToWorldTransform->at<double>(2,1) = -0.975717;
  modelToWorldTransform->at<double>(2,2) = -0.213327;
  modelToWorldTransform->at<double>(2,3) = -2010.96;

  modelToWorldTransform->at<double>(3,0) = 0.0;
  modelToWorldTransform->at<double>(3,1) = 0.0;
  modelToWorldTransform->at<double>(3,2) = 0.0;
  modelToWorldTransform->at<double>(3,3) = 1.0;

  Projector->SetModelToWorldTransform(modelToWorldTransform);
  Projector->SetModelPoints(ModelGridPoints);

  Projector->Project(matcher);
  MITK_TEST_CONDITION_REQUIRED (Projector->GetProjectOK(), "Testing mitkProjectPointsOnStereoVideo projected OK, when using model to world");

  MITK_TEST_CONDITION(CheckTransformedPointVector(Projector->GetPointsInLeftLensCS(), 4), "Testing projected points, when using model to world");

  Projector->ClearWorldPoints();
  Projector->ClearModelPoints();
  //these are the gold standard projected points for frame 1155
  //1155 664.844 69.984 753.793 68.306 628.092 279.283 711.968 279.424 1264.44 296.217 1365.82 296.783 1277.2 79.8817 1380.06 75.8718
  //these are perturbed world points that should yield the following errors for frame 1155
  WorldGridPoints.clear();
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-826.2000 ,  -207.2000 ,-2010.6000 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-820.4406 ,  -202.5256 , -2036.5725 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-820.5379 ,  -165.6142 , -2037.2575 )));
  WorldGridPoints.push_back(mitk::WorldPoint (cv::Point3d (-826.7656 ,  -167.0234 , -2008.2263 )));
  Projector->AppendWorldPoints(WorldGridPoints);
  Projector->AppendClassifierWorldPoints(WorldGridPoints);
  Projector->Project(matcher);

  std::vector < mitk::GoldStandardPoint> leftGS;
  std::vector < mitk::GoldStandardPoint> rightGS;
  leftGS.push_back(mitk::GoldStandardPoint (1154,-1, cv::Point2d(664.844, 69.984)));
  leftGS.push_back(mitk::GoldStandardPoint (1154,-1, cv::Point2d(628.092, 279.283)));
  leftGS.push_back(mitk::GoldStandardPoint (1154,-1, cv::Point2d(1264.44, 296.217)));
  leftGS.push_back(mitk::GoldStandardPoint (1154,-1, cv::Point2d(1277.2, 79.8817)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(753.793, 68.306)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(711.968, 279.424)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1365.82, 296.783)));
  rightGS.push_back(mitk::GoldStandardPoint (1155,-1, cv::Point2d(1380.06, 75.8718)));
  Projector->SetLeftGoldStandardPoints(leftGS, matcher);
  Projector->SetRightGoldStandardPoints(rightGS, matcher);
  Projector->CalculateProjectionErrors("",false);
  Projector->CalculateTriangulationErrors("");

  MITK_TEST_CONDITION(CheckProjectionErrors(Projector), "Testing projection Errors");
  MITK_TEST_CONDITION(CheckReProjectionErrors(Projector), "Testing re-projection Errors");
  MITK_TEST_CONDITION(CheckTriangulationErrors(Projector), "Testing triangulation Errors");
  MITK_TEST_CONDITION(CheckTriangulateGoldStandardPoints(Projector, matcher, argv[3]), "Testing whether triangulation of gold standard points works");
  //now we're going to repeat this test using more grid points, and a model to world transform
  Projector->ClearWorldPoints();
  Projector->SetModelToWorldTransform(modelToWorldTransform);

  ModelGridPoints->SetInOrderedMode(true);
  ModelGridPoints->AddPoint ( cv::Point3d (-1.416343, -0.452355, 0.051662), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (1.877316, 26.356847, -0.051924), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (38.706592, 28.911089, 0.051656), cv::Scalar ( 255,255,255));
  ModelGridPoints->AddPoint ( cv::Point3d (38.832543, -0.813564, -0.051916), cv::Scalar ( 255,255,255));

  Projector->SetModelPoints(ModelGridPoints);
  Projector->Project(matcher);
  Projector->CalculateProjectionErrors("",false);
  Projector->CalculateTriangulationErrors("");

  MITK_TEST_CONDITION(CheckProjectionErrors(Projector), "Testing projection Errors with model to world transform");
  MITK_TEST_CONDITION(CheckReProjectionErrors(Projector), "Testing re-projection Errors with model to world transform");
  MITK_TEST_CONDITION(CheckTriangulationErrors(Projector), "Testing triangulation Errors with model to world transform");

  MITK_TEST_END();
}
