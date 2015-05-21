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
#include <mitkOpenCVMaths.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <cmath>

/**
 * \file Tests for some of the functions in openCVMaths.
 */

void ArithmaticTests()
{
  cv::Point2d point1 = cv::Point2d ( 1.0 , 1.0 );
  cv::Point2d point2 = cv::Point2d ( 0.5 , 0.3 );
  cv::Point2d point3 = cv::Point2d ( 0.7 , 1.4 );
  cv::Point2d point4 = cv::Point2d ( 1.5 , 2.3 );

  double tolerance = 1e-6;
  MITK_TEST_CONDITION ( point1 == point1 , "Testing point2d equality operator" );
  MITK_TEST_CONDITION (mitk::NearlyEqual (point1, point1, tolerance) , "Testing Nearly Equal " );
  MITK_TEST_CONDITION (mitk::NearlyEqual (( point1 + point2 ), cv::Point2d ( 1.5 , 1.3 ), tolerance), "Testing addition operator");
  MITK_TEST_CONDITION (mitk::NearlyEqual ((point4 - point3) ,cv::Point2d(0.8, 0.9), tolerance), "Testing subtraction operator");

  cv::Point3d point3d_1 = cv::Point3d ( 1.0, 1.0, 1.0 );
  cv::Point3d point3d_2 = cv::Point3d ( 4.0, 3.0, -1.0 );
  cv::Matx44d mat44d_1;
  cv::Mat mat ( 4,4,CV_64FC1 );
  mat44d_1(0,0) = 1.0; mat44d_1(0,1) = 0.0; mat44d_1(0,2) = 0.0; mat44d_1(0,3) = 10.0;
  mat44d_1(1,0) = 0.0; mat44d_1(1,1) = 1.0; mat44d_1(1,2) = 0.0; mat44d_1(1,3) = 16.0;
  mat44d_1(2,0) = 0.0; mat44d_1(2,1) = 0.0; mat44d_1(2,2) = 1.0; mat44d_1(2,3) = 21.0;
  mat44d_1(3,0) = 0.0; mat44d_1(3,1) = 0.0; mat44d_1(3,2) = 0.0; mat44d_1(3,3) = 1.0;

  mat.at<double>(0,0) = 1.0; mat.at<double>(0,1) = 0.0; mat.at<double>(0,2) = 0.0; mat.at<double>(0,3) = 12.0;
  mat.at<double>(1,0) = 0.0; mat.at<double>(1,1) = 1.0; mat.at<double>(1,2) = 0.0; mat.at<double>(1,3) = 14.0;
  mat.at<double>(2,0) = 0.0; mat.at<double>(2,1) = 0.0; mat.at<double>(2,2) = 1.0; mat.at<double>(2,3) = 9.0;
  mat.at<double>(3,0) = 0.0; mat.at<double>(3,1) = 0.0; mat.at<double>(3,2) = 0.0; mat.at<double>(3,3) = 1.0;

  using namespace mitk;
  cv::Point3d product = mat * point3d_1;
  MITK_TEST_CONDITION (mitk::NearlyEqual(product, cv::Point3d(13.0, 15.0, 10.0), tolerance), "Checking cv::Mat * cv::Point3d operator");
  product = mat44d_1 * point3d_1;
  MITK_TEST_CONDITION (mitk::NearlyEqual(product, cv::Point3d(11.0, 17.0, 22.0), tolerance), "Checking cv::Matx44d * cv::Point3d operator");
  std::pair < cv::Point3d, cv::Point3d > pairProduct = mitk::TransformPointPair ( mat44d_1, 
      std::pair < cv::Point3d, cv::Point3d > ( point3d_1, point3d_2 ));
  MITK_TEST_CONDITION (mitk::NearlyEqual(pairProduct.first, cv::Point3d(11.0, 17.0, 22.0), tolerance), "Checking transform point pair first");
  MITK_TEST_CONDITION (mitk::NearlyEqual(pairProduct.second, cv::Point3d(14.0, 19.0, 20.0), tolerance), "Checking transform point pair second");

}

void RMSTest()
{
  //make some measured points
  std::vector < mitk::ProjectedPointPairsWithTimingError > measured;
  std::vector < mitk::ProjectedPointPairsWithTimingError > actual;

  mitk::ProjectedPointPairsWithTimingError measured_0;
  measured_0.m_Points.push_back(
      mitk::ProjectedPointPair (cv::Point2d ( -0.1, 0.7 ),cv::Point2d (1.1 , 0.9)));
  measured.push_back(measured_0);

  mitk::ProjectedPointPairsWithTimingError actual_0;
  actual_0.m_Points.push_back(
      mitk::ProjectedPointPair (cv::Point2d ( -0.1, 0.7 ),cv::Point2d (1.1 , 0.9)));
  actual.push_back(actual_0);

  int index = -1;
  cv::Point2d outlierSD = cv::Point2d(2.0, 2.0);
  long long allowableTimingError = 30e6;
  bool duplicateLines = true;
  std::pair <double,double> rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );

  double tolerance = 1e-4;
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d( rmsError.first, rmsError.second ), cv::Point2d ( 0.0, 0.0), tolerance) , "Testing RMSError returns 0.0 when no error" );

  mitk::ProjectedPointPairsWithTimingError actual_1;
  actual_1.m_Points.push_back(
      mitk::ProjectedPointPair (cv::Point2d ( -0.2, 0.5 ),cv::Point2d (1.2 , 0.3)));

  measured.push_back(measured_0);
  actual.push_back(actual_1);

  duplicateLines=false;
  rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d( rmsError.first, rmsError.second ) , cv::Point2d ( 0.15811, 0.43012), tolerance ) , "Testing RMSError returns right value for a real error" );
 
  duplicateLines=true;
  rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d (rmsError.first, rmsError.second) , cv::Point2d ( 0.0, 0.0), tolerance) , "Testing duplicate lines parameter has the desired effect" );

  mitk::ProjectedPointPairsWithTimingError measured_1;
  measured_1.m_Points.push_back(
      mitk::ProjectedPointPair (cv::Point2d ( -0.1, 0.7 ),cv::Point2d (1.1 , 0.9)));
  measured_1.m_TimingError = 30e7;
  measured.push_back(measured_1);
  actual.push_back(actual_0);

  duplicateLines=false;
  rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d (rmsError.first, rmsError.second ) ,cv::Point2d ( 0.15811, 0.43012), tolerance) , "Testing RMSError rejects high timing error points" );
 
  allowableTimingError = 31e7;

  outlierSD = cv::Point2d(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
  rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d(rmsError.first, rmsError.second) ,cv::Point2d ( 0.12910, 0.35119), tolerance) , "Testing RMSError accepts when allowable timing error increased" );
 
  outlierSD = cv::Point2d(2.0, 2.0);
  rmsError = mitk::RMSError ( measured , actual ,
      index, outlierSD, allowableTimingError , duplicateLines );
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( cv::Point2d(rmsError.first, rmsError.second) , cv::Point2d( 0.0, 0.0), tolerance) , "Testing RMSError culls outliers" );

}

void FindIntersectTest()
{
  cv::Vec4i line1;
  cv::Vec4i line2;
  cv::Point2d intersect;

  for ( unsigned int i = 0 ; i < 4 ; i ++ )
  {
    line1[i] = 0;
    line2[i] = 0;
  }
  intersect = mitk::FindIntersect (line1, line2);

  MITK_TEST_CONDITION ((boost::math::isnan(intersect.x) && boost::math::isnan(intersect.y)), "IntersectTest: all zeros : " << intersect );
  //parallel vertical lines
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 0 ; line1[3] = 1; 
  line2[0] = 1 ; line2[1] = 0 ; line2[2] = 1 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION ((boost::math::isnan(intersect.x) && boost::math::isnan(intersect.y)), "IntersectTest: parallel vertical lines" << intersect );
  //parallel horizontal lines
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 1 ; line1[3] = 0; 
  line2[0] = 0 ; line2[1] = 1 ; line2[2] = 1 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION ((boost::math::isnan(intersect.x) && boost::math::isnan(intersect.y)), "IntersectTest: parallel horizontal lines" << intersect );
  //parallel oblique lines
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 1 ; line1[3] = 1; 
  line2[0] = 1 ; line2[1] = 0 ; line2[2] = 2 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION ((boost::math::isnan(intersect.x) && boost::math::isnan(intersect.y)), "IntersectTest: parallel oblique lines" << intersect );
  //simple intersection
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 1 ; line1[3] = 1; 
  line2[0] = 1 ; line2[1] = 0 ; line2[2] = 0 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION (intersect == cv::Point2d(0.5,0.5), "IntersectTest: 45 degrees intersect at 0.5" << intersect );
  //line1 vertical
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 0 ; line1[3] = 1; 
  line2[0] = 1 ; line2[1] = 0 ; line2[2] = 0 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION (intersect == cv::Point2d(0.0,1.0), "IntersectTest: line 1 vertical " << intersect );
  //line 2 vertical
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 1 ; line1[3] = 1; 
  line2[0] = 1 ; line2[1] = 0 ; line2[2] = 1 ; line2[3] = 1;
  intersect = mitk::FindIntersect (line1, line2);
  MITK_TEST_CONDITION (intersect == cv::Point2d(1.0,1.0), "IntersectTest: line 2 vertical " << intersect );
  
}

void FindIntersectsTest()
{
  cv::Vec4i line1;
  cv::Vec4i line2;
  cv::Vec4i line3;
  cv::Vec4i line4;

  //line1 is degenerate
  line1[0] = 0 ; line1[1] = 0 ; line1[2] = 0 ; line1[3] = 0;
  //line2 y = 2x + 10
  line2[0] = 0 ; line2[1] = 10 ; line2[2] = 10 ; line2[3] = 30;
  //line3 y = -x ;
  line3[0] = -5; line3[1] = 5 ; line3[2] = 5; line3[3] = -5;
  //line4 y = x;
  line4[0] = 0; line4[1] = 0 ; line4[2] = 15; line4[3] = 15;

  std::vector <cv::Vec4i> lines;
  lines.push_back(line1);
  lines.push_back(line2);
  lines.push_back(line3);
  lines.push_back(line4);

  bool rejectPointsNotOnBothLines = false;
  bool rejectNonPerpendicularLines = false;
  double perpendicularityTolerance = 20.0;
  
  std::vector <cv::Point2d> intersects;
  intersects = mitk::FindIntersects ( lines, rejectPointsNotOnBothLines, rejectNonPerpendicularLines, perpendicularityTolerance );
  // there should be 3 intersections, (-10/3, 10/3 ) , (-10,-10), and ( 0,0)
  MITK_TEST_CONDITION ( intersects.size() == 3 , "Testing size of intesects vector with no conditions " << intersects.size());
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[0], cv::Point2d (-3.333333, 3.333333), 1e-6) , "Testing first intersect with no conditions " << intersects[0]);
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[1], cv::Point2d (-10.0, -10.0),1e-6) , "Testing second intersect with no conditions " << intersects[1]);
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[2], cv::Point2d (0.0, 0.0), 1e-6) , "Testing third intersect with no conditions " << intersects[2]);
  // add a perpendicularity constraint should remove line 2 line 4
  rejectNonPerpendicularLines = true;
  perpendicularityTolerance = 20.0;
  intersects = mitk::FindIntersects ( lines, rejectPointsNotOnBothLines, rejectNonPerpendicularLines, perpendicularityTolerance );
  MITK_TEST_CONDITION ( intersects.size() == 2 , "Testing size of intesects vector with 20 degree perpendicularity" << intersects.size());
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[0], cv::Point2d (-3.333333, 3.333333), 1e-6) , "Testing first intersect with no conditions " << intersects[0]);
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[1], cv::Point2d (0.0, 0.0), 1e-6) , "Testing second intersect with no conditions " << intersects[1]);
  // add an on both lines constraint should leave only line3 line4
  rejectPointsNotOnBothLines = true;
  intersects = mitk::FindIntersects ( lines, rejectPointsNotOnBothLines, rejectNonPerpendicularLines, perpendicularityTolerance );
  MITK_TEST_CONDITION ( intersects.size() == 1 , "Testing size of intesects vector with 20 degree perpendicularity and on interval requirement" << intersects.size());
  MITK_TEST_CONDITION ( mitk::NearlyEqual ( intersects[0], cv::Point2d (0.0, 0.0), 1e-6) , "Testing first intersect with all conditions " << intersects[0]);


}
void AngleBetweenLinesTest()
{
  double tolerance = 1e-6;
  cv::Vec4i line1;
  cv::Vec4i line2;

  for ( unsigned int i = 0 ; i < 4 ; i++ )
  {
    line1[i] = 0;
    line2[i] = 0;
  }

  MITK_TEST_CONDITION ( boost::math::isnan(mitk::AngleBetweenLines ( line1, line2)) , "Testing angle between 2 zero vectors " << mitk::AngleBetweenLines ( line1, line2) );

  line1[1] = 1;
  line2[3] = 1;

  MITK_TEST_CONDITION ( ( fabs (mitk::AngleBetweenLines ( line1, line2) - 0.0) < tolerance  ) , "Testing angle between 2 parallel vectors " << mitk::AngleBetweenLines ( line1, line2) );

  line2[1] = 1;
  line2[0] = 1;
  
  MITK_TEST_CONDITION ( ( fabs (mitk::AngleBetweenLines ( line1, line2) - CV_PI/2.0 ) < tolerance ) , "Testing angle between 2 perpendicular vectors " << mitk::AngleBetweenLines ( line1, line2) );

  line2[1]=2;
  
  MITK_TEST_CONDITION ( ( fabs (mitk::AngleBetweenLines ( line1, line2) - CV_PI/4.0) < tolerance ) , "Testing angle between 2 vectors at 45 degrees " << mitk::AngleBetweenLines ( line1, line2) );

  for ( unsigned int i = 0 ; i < 4 ; i++ )
  {
    line1[i] = 0;
    line2[i] = 0;
  }

  line1[0] = 1;
  line2[2] = -1;

  MITK_TEST_CONDITION ( ( fabs (mitk::AngleBetweenLines ( line1, line2) - 0.0 ) < tolerance ) , "Testing angle between 2 opposite parallel vectors " << mitk::AngleBetweenLines ( line1, line2) );

}

void CheckIfLinesArePerpendicularTest ( )
{
 
  double angleTolerance = 10.0;
  cv::Vec4i line1;
  cv::Vec4i line2;

  for ( unsigned int i = 0 ; i < 4 ; i++ )
  {
    line1[i] = 0;
    line2[i] = 0;
  }

  MITK_TEST_CONDITION ( (! mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 zero vectors are not perpendicular" );

  line1[1] = 1;
  line2[3] = 1;

  MITK_TEST_CONDITION ( (! mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance) ), "Checking that 2 parallel vectors are not perpendicular" );

  angleTolerance = 90.0;
  
  MITK_TEST_CONDITION ( ( mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance) ), "Checking that 2 parallel vectors are perpendicular within 90 degrees tolerance" );
  
  line1[1] = 0;
  line2[3] = 0;
  
  MITK_TEST_CONDITION ( (! mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 zero vectors are not perpendicular even when tolerance = 90 degrees" );
  
  angleTolerance = 0.0;

  line1[0] = 1000;
  line2[1] = -1000;
  
  MITK_TEST_CONDITION ( ( mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 perpendicular vectors are exactly  perpendicular" );

  angleTolerance = 10.0;
  line1[1]=-167;
  
  
  MITK_TEST_CONDITION ( ( mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 lines at 80.5 degrees are perpendicular within 10 degrees" );

  angleTolerance = 9.0;
  
  MITK_TEST_CONDITION ( ( ! mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 lines at 80.5 degrees are not perpendicular within 9 degrees" );
  
  angleTolerance = 10.0;
  line1[1]=167;
  
  MITK_TEST_CONDITION ( ( mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 lines at 99.5 degrees are perpendicular within 10 degrees" );

  angleTolerance = 9.0;
  
  MITK_TEST_CONDITION ( ( ! mitk::CheckIfLinesArePerpendicular(line1,line2, angleTolerance ) ), "Checking that 2 lines at 99.5 degrees are not perpendicular within 9 degrees" );
  
}


void PointInIntervalTest ( )
{
 
  cv::Vec4i line;

  //when x2 and y2 are greater than x1 and x1

  line[0] = -10;
  line[1] = 5;
  line[2] = 5; 
  line[3] = 7;

  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(-10.1,6),line)), "Point just outside x bound" );
  MITK_TEST_CONDITION ( ( mitk::PointInInterval(cv::Point2d(-10.0,5),line)), "Point on x bound" );
  MITK_TEST_CONDITION ( ( mitk::PointInInterval(cv::Point2d(-9.9,5.5),line)), "Point just inside x bound" );
  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(-10.0,0),line)), "Point outside y bound" );
  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(20.0,0),line)), "Point outside both bound" );

  //when x1 and y1 are greater than x2 and x2

  line[2] = -10;
  line[3] = 5;
  line[0] = 5; 
  line[1] = 7;

  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(-10.1,6),line)), "Point just outside x bound" );
  MITK_TEST_CONDITION ( ( mitk::PointInInterval(cv::Point2d(-10.0,5),line)), "Point on x bound" );
  MITK_TEST_CONDITION ( ( mitk::PointInInterval(cv::Point2d(-9.9,5.5),line)), "Point just inside x bound" );
  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(-10.0,0),line)), "Point outside y bound" );
  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(20.0,0),line)), "Point outside both bound" );
  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(std::numeric_limits<double>::quiet_NaN(),0),line)), "Point is NaN" );

  line[2] = 0;
  line[3] = 0;
  line[0] = 0; 
  line[1] = 0;

  MITK_TEST_CONDITION ( ( ! mitk::PointInInterval(cv::Point2d(-10.1,6),line)), "Zero interval" );
  MITK_TEST_CONDITION ( ( mitk::PointInInterval(cv::Point2d(0.0,0.0),line)), "Zero interval and point" );
 
}

void DistanceToLineTest ()
{
  cv::Point3d u ( 0.0,0.0,0.0);
  cv::Point3d v ( 1.0,1.0,1.0);
  cv::Point3d x ( 0.0,1.0,0.5);
  
  double dtl = mitk::DistanceToLine (std::pair<cv::Point3d, cv::Point3d>(u,v),x);
  MITK_TEST_CONDITION( fabs(dtl - 0.70711) < 1e-5 ,"Checking distance to line " << dtl);

  cv::Point3d u1 ( 10.0, 10.0 , 10.0);
  cv::Point3d v1 ( 11.0, 11.0, 11.0);

  double dtl1 = mitk::DistanceToLine (std::pair<cv::Point3d, cv::Point3d>(u1,v1),x);
  MITK_TEST_CONDITION( fabs(dtl1 - 0.70711) < 1e-5 ,"Checking distance to line again " << dtl1);
}

void DistanceToLineSegmentTest ()
{
  cv::Point3d u ( 0.0,0.0,0.0);
  cv::Point3d v ( 1.0,1.0,1.0);
  cv::Point3d x ( 0.0,1.0,0.5);
  
  double dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(u,v),x);
  MITK_TEST_CONDITION( fabs(dtl - 0.70711) < 1e-5 ,"Checking distance to line segment with projection in segment: " << dtl);
  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(v,u),x);
  MITK_TEST_CONDITION( fabs(dtl - 0.70711) < 1e-5 ,"Checking distance to line segment with projection in segment, flipped ends: " << dtl);

  x = cv::Point3d ( 2.0, 2.0, 2.0 );
  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(u,v),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(3.0)) < 1e-5 ,"Checking distance to line segment with projection on line but off segment: " << dtl);
  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(v,u),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(3.0)) < 1e-5 ,"Checking distance to line segment with projection on line but off segment, flipped ends: " << dtl);

  v = cv::Point3d ( -1.0, -1.0, -1.0 );
  x = cv::Point3d ( 0.0,1.0,0.5);

  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(u,v),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(1.25)) < 1e-5 ,"Checking distance to line segment with projection off line and off segment: " << dtl);
  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(v,u),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(1.25)) < 1e-5 ,"Checking distance to line segment with projection off line and off segment, flipped ends: " << dtl);

  u = cv::Point3d ( -2.0, -2.0, -2.0 );
  x = cv::Point3d ( 0.0,1.0,-0.5);

  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(u,v),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(5.25)) < 1e-5 ,"Checking distance to line segment with projection off line and off segment: " << dtl);
  dtl = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(v,u),x);
  MITK_TEST_CONDITION( fabs(dtl - sqrt(5.25)) < 1e-5 ,"Checking distance to line segment with projection off line and off segment, flipped ends: " << dtl);

  cv::Point3d u1 ( 10.0, 10.0 , 10.0);
  cv::Point3d v1 ( 11.0, 11.0, 11.0);
  x = cv::Point3d ( 0.0,1.0,0.5);

  double dtl1 = mitk::DistanceToLineSegment (std::pair<cv::Point3d, cv::Point3d>(u1,v1),x);
  MITK_TEST_CONDITION( fabs(dtl1 - 16.4696690) < 1e-5 ,"Checking point off line and off segment " << dtl1);
}


void DotAndCrossProductTest ()
{
  cv::Point3d x ( 2.0,3.0,5.0);
  cv::Point3d y ( 7.0,11.0,13.0);
  cv::Point3d cp (-16.0,9.0,1.0);
  double dp =  112.0;

  MITK_TEST_CONDITION( mitk::NearlyEqual (mitk::CrossProduct(x,y),cp,1e-6),"Checking cross product " << mitk::CrossProduct(x,y));
  MITK_TEST_CONDITION( ( mitk::DotProduct(x,y) - dp < 1e-6),"Checking dot product " << mitk::DotProduct(x,y));
}

void NormTest ()
{
  cv::Point3d x ( 2.0,3.0,5.0);
  
  MITK_TEST_CONDITION ( ( mitk::Norm (x) - 6.1644)  < 1e-4, "Checking point vector norm " << mitk::Norm(x) );
}

void DistanceBetweenLinesTest ()
{
  //a line through the origin an (1,1,0) 
  cv::Point3d P0 (0,0,0);
  cv::Point3d u (0.70711,0.7011,0);
  //a line through (1,0,1) and (0,1,1)
  cv::Point3d Q0 (1,0,1);
  cv::Point3d v (-0.70711,0.7011,0);
  cv::Point3d midPoint;
  double distance = mitk::DistanceBetweenLines(P0, u, Q0 , v , midPoint);
  
  MITK_TEST_CONDITION ( ( distance - 1.0 )  < 1e-6, "Checking distance between two lines " << distance );
  MITK_TEST_CONDITION( mitk::NearlyEqual (midPoint,cv::Point3d(0.5,0.5,0.5),1e-6),"Checking midpoint " << midPoint);
}

void TwoPointsToPLambdaTest ()
{
  //a line through the origin an (1,1,1) 
  cv::Point3d P0 (0,0,0);
  cv::Point3d P1 (1,1,1);
  //a line through (1,1,1) and (-3,1,-3)
  cv::Point3d Q0 (1,1,1);
  cv::Point3d Q1 (-3,1,-3);

  cv::Point3d x1;
  cv::Point3d u1;
  cv::Point3d x2;
  cv::Point3d u2;

  std::pair <cv::Point3d, cv::Point3d>  line1 = mitk::TwoPointsToPLambda (std::pair <cv::Point3d, cv::Point3d> ( P0,P1));
  std::pair <cv::Point3d, cv::Point3d>  line2 = mitk::TwoPointsToPLambda (std::pair <cv::Point3d, cv::Point3d> ( Q0,Q1));
  
  MITK_TEST_CONDITION( mitk::NearlyEqual (line1.first,cv::Point3d(0,0,0),1e-6),"Checking line 1 point " << line1.first);
  MITK_TEST_CONDITION( mitk::NearlyEqual (line1.second,cv::Point3d(-1/sqrt(3.0),-1/sqrt(3.0),-1/sqrt(3.0)),1e-6),"Checking line 1 vector " << line1.second);
  MITK_TEST_CONDITION( mitk::NearlyEqual (line2.first,cv::Point3d(1,1,1),1e-6),"Checking line 2 point " << line2.first);
  MITK_TEST_CONDITION( mitk::NearlyEqual (line2.second,cv::Point3d(4/sqrt(32.0),0,4/sqrt(32.0)),1e-6),"Checking line 2 vector " << line2.second);
}

void RemoveOutliersTest ()
{
  std::vector < cv::Point3d > points;
  unsigned int pointsGone;

  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 0 , " Remove points test with empty vector " << pointsGone ); 

  points.push_back ( cv::Point3d ( std::numeric_limits<double>::infinity () , 0 , 0 ) );
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 0 , " Remove points test with infinite point and infinite limit " << pointsGone ); 
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), 10, 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 1 , " Remove points test with infinite point and finite limit " << pointsGone ); 
  points.push_back ( cv::Point3d ( 0,  std::numeric_limits<double>::quiet_NaN () , 0 ) );
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 0 , " Remove points test with NaN point and infinite limit " << pointsGone ); 
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity() , 
      -(std::numeric_limits<double>::infinity()), 10, 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 0 , " Remove points test with NaN point and finite limit " << pointsGone ); 
  points.push_back ( cv::Point3d ( 0,  -20.0, 50.0 ) );
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity() , 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::quiet_NaN (), 
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
  MITK_TEST_CONDITION ( pointsGone == 0 , " Remove points test with NaN point and NaN limit " << pointsGone ); 
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity() , 
      -(std::numeric_limits<double>::infinity()), 0.0, 
      -(std::numeric_limits<double>::infinity()), 40.0);
  MITK_TEST_CONDITION ( pointsGone == 1 , " Remove points test with numeric point and numeric limit " << pointsGone ); 
  points.push_back ( cv::Point3d ( 0,  -20.0, 50.0 ) );
  pointsGone= mitk::RemoveOutliers ( points ,
      -(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity() , 
      -10.0, 0.0, 
      -(std::numeric_limits<double>::infinity()), 100.0);
  MITK_TEST_CONDITION ( pointsGone == 1 , " Remove points test with numeric point and numeric limit " << pointsGone ); 
}

void FindNearestPointTest ()
{
  mitk::PickedObject p ( "left", 0, 0 );
  std::vector < mitk::PickedObject> classifierPoints;
  mitk::PickedObject c1( "left", 0, 0 );
  c1.m_Id = 0;
  c1.m_Points.push_back ( cv::Point3d ( 1.0, 0.0, 0.0 ));
  classifierPoints.push_back ( c1 );
  mitk::PickedObject c2( "left", 0, 0 );
  c2.m_Id = 1;
  c2.m_Points.push_back ( cv::Point3d ( 1.0, 1.0, 0.0 ));
  classifierPoints.push_back ( c2 );
  mitk::PickedObject c3( "left", 0, 0 );
  c3.m_Id = 2;
  c3.m_Points.push_back ( cv::Point3d ( 0.0, 1.0, 0.0 ));
  classifierPoints.push_back ( c3 );
  mitk::PickedObject c4( "left", 0, 0 );
  c4.m_Id = 3;
  c4.m_Points.push_back ( cv::Point3d ( 0.0, 0.0, 0.0 ));
  classifierPoints.push_back ( c4 );
  mitk::PickedObject c5( "left", 0, 0 );
  c5.m_Id = 0;
  c5.m_IsLine = true;
  c5.m_Points.push_back ( cv::Point3d ( 0.0, 0.0, 0.0 ));
  c5.m_Points.push_back ( cv::Point3d ( 1.0, 1.0, 0.0 ));
  classifierPoints.push_back ( c5 );
  
  p.m_Points.push_back ( cv::Point3d ( 0.3, 0.6, 0.0) );
  p.m_Id = -1;

  mitk::PickedObject matched;
  double minRatio = 0 ;
  matched = mitk::FindNearestPoint ( p , classifierPoints, &minRatio );

  MITK_TEST_CONDITION ( matched.m_Id == 2 , "Testing matched to point 2, got , " << matched.m_Id );
  MITK_TEST_CONDITION ( ( fabs ( minRatio - 0.6708/0.5000)) < 1e-3  , "Testing min ratio is 1.342 , got  " << minRatio) ;

  p.m_Id = 3;
  matched = mitk::FindNearestPoint ( p , classifierPoints, &minRatio );
  MITK_TEST_CONDITION ( matched.m_Id == 3 , "Testing matched to point 3 when point id set, got , " << matched.m_Id );
  MITK_TEST_CONDITION ( boost::math::isinf(minRatio), "Testing min ratio is infinite , got  " << minRatio) ;

  p.m_IsLine = true;
  matched = mitk::FindNearestPoint ( p , classifierPoints, &minRatio );
  MITK_TEST_CONDITION ( matched.m_Id == -1 , "Testing matched to nothing (-1) when point id beyond end, got , " << matched.m_Id );
  MITK_TEST_CONDITION ( boost::math::isnan(minRatio), "Testing min ratio is NaN , got  " << minRatio) ;

  p.m_Id = -1;
  matched = mitk::FindNearestPoint ( p , classifierPoints, &minRatio );
  MITK_TEST_CONDITION ( matched.m_Id == 0 , "Testing matched to line 0 when point id not set, got , " << matched.m_Id );
  MITK_TEST_CONDITION ( boost::math::isinf(minRatio), "Testing min ratio is inf , got  " << minRatio) ;

  classifierPoints.clear();
  matched = mitk::FindNearestPoint ( p , classifierPoints, &minRatio );
  MITK_TEST_CONDITION ( matched.m_Id == -1 , "Testing matched to nothing (-1) when classifier points in empty " << matched.m_Id );
  MITK_TEST_CONDITION ( boost::math::isnan(minRatio), "Testing min ratio is NaN , got  " << minRatio) ;
  
}

void DistanceBetweenTwoPointsTest()
{
  cv::Point3d p1 ( -1.1, 20.2, 3.0);
  cv::Point3d p2 ( 1.1, 10.4, 3.6);
  cv::Point3d p3 ( 1.1, 10.4, std::numeric_limits<double>::quiet_NaN());
  cv::Point3d p4 ( 1.1, 10.4, std::numeric_limits<double>::infinity());

  double distance;
  distance = mitk::DistanceBetweenTwoPoints ( p1, p2 );
  MITK_TEST_CONDITION ( fabs ( distance - 10.062 ) < 1e-3 , "Testing distance between points = 10.062, got : " << distance );
  distance = mitk::DistanceBetweenTwoPoints ( p2, p1 );
  MITK_TEST_CONDITION ( fabs ( distance - 10.062 ) < 1e-3 , "Testing distance between points = 10.062, got : " << distance );
  distance = mitk::DistanceBetweenTwoPoints ( p1, p3 );
  MITK_TEST_CONDITION ( boost::math::isnan( distance ), "Testing distance between points is not a number, got : " << distance );
  distance = mitk::DistanceBetweenTwoPoints ( p2, p4 );
  MITK_TEST_CONDITION ( boost::math::isinf( distance ), "Testing distance between points is infinite, got : " << distance );
  distance = mitk::DistanceBetweenTwoPoints ( p4, p3 );
  MITK_TEST_CONDITION ( boost::math::isnan( distance ), "Testing distance between points is not a number, got : " << distance );

}

void DistanceBetweenTwoSplinesTest()
{}

int mitkOpenCVMathTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVMathTests");

  //MITK_TEST_CONDITION ( ArithmaticTests() , "Testing basic arithmetic");
  //MITK_TEST_CONDITION ( RMSTest(), "Testing RMSError" );
  ArithmaticTests();
  RMSTest();
  FindIntersectTest();
  FindIntersectsTest();
  AngleBetweenLinesTest();
  CheckIfLinesArePerpendicularTest();
  PointInIntervalTest();
  DistanceToLineTest();
  DistanceToLineSegmentTest();
  DotAndCrossProductTest();
  NormTest();
  DistanceBetweenLinesTest();
  TwoPointsToPLambdaTest ();
  RemoveOutliersTest();
  FindNearestPointTest();
  DistanceBetweenTwoPointsTest();
  DistanceBetweenTwoSplinesTest();

  MITK_TEST_END();
}



