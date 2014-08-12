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

  MITK_TEST_CONDITION ( point1.LeftNaN() == true, "Testing leftNaN handling 1" ) ;
  MITK_TEST_CONDITION ( point1.RightNaN() == false, "Testing rightNaN handling 1" ) ;

  MITK_TEST_CONDITION ( point2.LeftNaN() == true, "Testing leftNaN handling 2" ) ;
  MITK_TEST_CONDITION ( point2.RightNaN() == true, "Testing rightNaN handling 2" ) ;

  MITK_TEST_CONDITION ( point3.LeftNaN() == false, "Testing leftNaN handling 3" ) ;
  MITK_TEST_CONDITION ( point3.RightNaN() == true, "Testing rightNaN handling 3" ) ;

  MITK_TEST_CONDITION ( point4.LeftNaN() == false, "Testing leftNaN handling 4" ) ;
  MITK_TEST_CONDITION ( point4.RightNaN() == false, "Testing rightNaN handling 4" ) ;

  mitk::WorldPoint x;
  MITK_TEST_END();
}


