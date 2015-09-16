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
#include <cmath>

/**
 * \file Test mitkOpenCVPointTypes
 */

void TestPickedObjectCompare()
{
  mitk::PickedObject p1;
  mitk::PickedObject p2;

  MITK_TEST_CONDITION ( p1.HeadersMatch(p2) , "Testing headers match for empty point list" );

  p1.m_Id = 2;

  MITK_TEST_CONDITION ( ! p2.HeadersMatch(p1) , "Testing headers don't match for different ID" );

  p2.m_Id =2;
  p2.m_IsLine = true;

  MITK_TEST_CONDITION ( ! p2.HeadersMatch(p1) , "Testing headers don't match for different isLine" );

  p1.m_IsLine = true;
  p1.m_FrameNumber = 200;

  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing headers don't match for different framenumbers" );
  
  p2.m_FrameNumber = 200;
  p2.m_Channel = "left";
  
  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing headers don't match for different channels" );

  p1.m_Channel = "left";

  MITK_TEST_CONDITION ( p1.HeadersMatch(p2) , "Testing headers do match" );

  p1.m_Id = -1;

  MITK_TEST_CONDITION ( ! p1.HeadersMatch(p2) , "Testing wild card for p1 doesn't  match" );
  MITK_TEST_CONDITION ( p2.HeadersMatch(p1) , "Testing wild card for p2 does match" );
  
  p1.m_TimeStamp = 100;
  MITK_TEST_CONDITION ( ! p2.HeadersMatch (p1, 10) , "Testing timing error check works - no match" );
  MITK_TEST_CONDITION ( p2.HeadersMatch (p1, 101) , "Testing timing error check works - match" );


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
  MITK_TEST_END();
}


