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
#include <mitkOpenCVPointTypes.h>
#include <mitkOpenCVMaths.h>
#include <cmath>

/**
 * \file Test mitkOpenCVPointTypes
 */

void TestPickedObjectCompare()
{
  mitk::PickedObject p1;
  mitk::PickedObject p2;

  MITK_TEST_CONDITION ( p1.HeadersMatch(p2) , "Testing headers match for empty point list" );
  MITK_TEST_CONDITION ( (! (p1 < p2)) && (! (p2 < p1)), "Testing < operator for empty picked object");

  p1.m_Id = 2;

  MITK_TEST_CONDITION ( ! p2.HeadersMatch(p1) , "Testing headers don't match for different ID" );
  MITK_TEST_CONDITION ( p2 < p1 , "Testing < operator for one point with id 2");

  p2.m_Id =2;
  p2.m_IsLine = true;

  MITK_TEST_CONDITION ( ! p2.HeadersMatch(p1) , "Testing headers don't match for different isLine" );
  MITK_TEST_CONDITION ( p1 < p2 , "Testing < operator for points before lines");

  p1.m_IsLine = true;
  p1.m_FrameNumber = 200;

  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing headers don't match for different framenumbers" );
  MITK_TEST_CONDITION ( p2 < p1 , "Testing < operator for frame number = 200");

  p2.m_FrameNumber = 200;
  p2.m_Channel = "left";

  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing headers don't match for different channels" );

  p1.m_Channel = "left";

  MITK_TEST_CONDITION ( p1.HeadersMatch(p2) , "Testing headers do match" );
  MITK_TEST_CONDITION ( (! (p1 < p2)) && (! (p2 < p1)), "Testing < operator for both frame numbers = 200, and two left channels");

  p1.m_Id = -1;

  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing wild card for p1 doesn't  match" );
  MITK_TEST_CONDITION ( p2.HeadersMatch(p1) , "Testing wild card for p2 does match" );

  p1.m_TimeStamp = 100;
  MITK_TEST_CONDITION ( ! p2.HeadersMatch (p1, 10) , "Testing timing error check works - no match" );
  MITK_TEST_CONDITION ( p2.HeadersMatch (p1, 101) , "Testing timing error check works - match" );

}

void TestPickedObjectMultipy()
{
  mitk::PickedObject p1;
  cv::Mat* mat = new cv::Mat(4,4,CV_64FC1);

  for ( unsigned int i = 0 ; i < 3 ; ++i )
  {
    for ( unsigned int j = 0 ; j < 3 ; ++j )
    {
      if ( i != j )
      {
        mat->at<double>(i,j) = 0.0;
      }
      else
      {
        mat->at<double>(i,j) = 1.0;
      }
    }
  }
  mat->at<double>(0,3) = 20;
  mat->at<double>(1,3) = -30;
  mat->at<double>(2,3) = 0;
  mat->at<double>(3,3) = 1.0;

  p1.m_Points.push_back ( cv::Point3d ( 0.0, 0.0, 0.0 ) );
  p1.m_Points.push_back ( cv::Point3d ( 1.0, 0.0, 5.0 ) );
  p1.m_Points.push_back ( cv::Point3d ( 0.0, 100.0, 0.0 ) );

  mitk::PickedObject p2 = p1 * mat;
  MITK_TEST_CONDITION ( p2.HeadersMatch ( p1, 1 ), "Testing that header for multiplied picked objects matches input");
  MITK_TEST_CONDITION ( mitk::NearlyEqual(p2.m_Points[0],cv::Point3d ( 20.0, -30.0 ,0.0 ), 1e-6), "Testing value of multiplied picked object 0 " << p2.m_Points[0]);
  MITK_TEST_CONDITION ( mitk::NearlyEqual(p2.m_Points[1],cv::Point3d ( 21.0, -30.0 ,5.0 ), 1e-6), "Testing value of multiplied picked object 1 " << p2.m_Points[1]);
  MITK_TEST_CONDITION ( mitk::NearlyEqual(p2.m_Points[2],cv::Point3d ( 20.0, 70.0 , 0.0 ), 1e-6), "Testing value of multiplied picked object 2 " << p2.m_Points[2]);
}

int mitkOpenCVPointTypesTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkPointTypesTest");

  mitk::ProjectedPointPair point1 = mitk::ProjectedPointPair (
      cv::Point2d ( std::numeric_limits<double>::quiet_NaN(), 0.0),
      cv::Point2d ( 0.0, 0.0));
  mitk::ProjectedPointPair point2 = mitk::ProjectedPointPair ();
  mitk::ProjectedPointPair point3 = mitk::ProjectedPointPair (
            cv::Point2d ( std::numeric_limits<double>::infinity(), 0.0),
            cv::Point2d ( std::numeric_limits<double>::quiet_NaN(), 0.0));
  mitk::ProjectedPointPair point4 = mitk::ProjectedPointPair (
            cv::Point2d ( 1.0, 0.0),
            cv::Point2d ( 0.0, 0.0));

  MITK_TEST_CONDITION ( point1.LeftNaNOrInf() == true, "Testing leftNaN handling 1" ) ;
  MITK_TEST_CONDITION ( point1.RightNaNOrInf() == false, "Testing rightNaN handling 1" ) ;

  MITK_TEST_CONDITION ( point2.LeftNaNOrInf() == true, "Testing leftNaN handling 2" ) ;
  MITK_TEST_CONDITION ( point2.RightNaNOrInf() == true, "Testing rightNaN handling 2" ) ;

  MITK_TEST_CONDITION ( point3.LeftNaNOrInf() == true, "Testing leftNaN handling 3" ) ;
  MITK_TEST_CONDITION ( point3.RightNaNOrInf() == true, "Testing rightNaN handling 3" ) ;

  MITK_TEST_CONDITION ( point4.LeftNaNOrInf() == false, "Testing leftNaN handling 4" ) ;
  MITK_TEST_CONDITION ( point4.RightNaNOrInf() == false, "Testing rightNaN handling 4" ) ;

  mitk::WorldPoint x;

  TestPickedObjectCompare();
  TestPickedObjectMultipy();
  MITK_TEST_END();
}


