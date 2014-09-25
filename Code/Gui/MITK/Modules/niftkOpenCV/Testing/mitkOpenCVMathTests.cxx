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
#include <cmath>

/**
 * \file Tests for some of the functions in openCVMaths.
 */

bool ArithmaticTests()
{
  MITK_TEST_BEGIN ("mitkOpenCVMathArithmaticTest");

  cv::Point2d point1 = cv::Point2d ( 1.0 , 1.0 );
  cv::Point2d point2 = cv::Point2d ( 0.5 , 0.3 );
  cv::Point2d point3 = cv::Point2d ( 0.7 , 1.4 );
  cv::Point2d point4 = cv::Point2d ( 1.5 , 2.3 );

  double tolerance = 1e-6;
  MITK_TEST_CONDITION ( point1 == point1 , "Testing point2d equality operator" );
  MITK_TEST_CONDITION (mitk::NearlyEqual (point1, point1, tolerance) , "Testing Nearly Equal " );
  MITK_TEST_CONDITION (mitk::NearlyEqual (( point1 + point2 ), cv::Point2d ( 1.5 , 1.3 ), tolerance), "Testing addition operator");
  MITK_TEST_CONDITION (mitk::NearlyEqual ((point4 - point3) ,cv::Point2d(0.8, 0.9), tolerance), "Testing subtraction operator");
  MITK_TEST_END();
}

bool RMSTest()
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVMathRMSTest");

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

  MITK_TEST_END();
}

bool FindIntersectTest()
{
  MITK_TEST_BEGIN(mitkFindInterectTest);

  MITK_TEST_CONDITION ( false, "implement this test" );

  MITK_TEST_END()
}

int mitkOpenCVMathTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVMathTests");

  //MITK_TEST_CONDITION ( ArithmaticTests() , "Testing basic arithmetic");
  //MITK_TEST_CONDITION ( RMSTest(), "Testing RMSError" );
  ArithmaticTests();
  RMSTest();
  FindIntersectTest();
  MITK_TEST_END();
}



