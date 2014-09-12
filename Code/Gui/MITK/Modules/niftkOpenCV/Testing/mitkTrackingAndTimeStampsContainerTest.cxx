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
#include <mitkTrackingAndTimeStampsContainer.h>

/**
 * \file Test harness for mitk::TrackingAndTimeStampsContainer.
 */

cv::Matx44d MakeMatrix ( double xrot, double xtrans)
{
  cv::Matx44d matrix;
  double theta = xrot * 3.1415926535897932 / 180;
  for ( int row = 0 ; row < 4 ; row ++ ) 
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      matrix(row,col) = 0.0;
    }
  }
  matrix(0,0) = 1.0;
  matrix (1,1) = cos(theta);
  matrix (2,2) = cos(theta);
  matrix (1,2) = - sin (theta);
  matrix (2,1) = sin(theta);
  matrix (0,3) = xtrans;
  matrix (3,3) = 1.0;
  return matrix;

}
bool CompareMatrices ( cv::Matx44d mat1 , cv::Matx44d mat2 )
{
  double precision = 1e-6;
  double error = 0.0;
  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      error += fabs(mat1(row,col) - mat2(row,col));
    }
  }
  if ( error < precision)
  {
    return true;
  }
  else
  {
    return false;
  }
}
int mitkTrackingAndTimeStampsContainerTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkTrackingAndTimeStampsContainerTest");

  mitk::TrackingAndTimeStampsContainer trackingAndTimeStamps;

  //lets test the iterate matrices function
  mitk::TimeStampsContainer::TimeStamp first = 10;
  mitk::TimeStampsContainer::TimeStamp second = 20;
  mitk::TimeStampsContainer::TimeStamp timingError;
  cv::Matx44d firstMatrix;
  cv::Matx44d secondMatrix;

  //first matrix is a 10 degree rotation around the x axis and a small translation
  firstMatrix = MakeMatrix ( 10, 50 ) ;
  //second matrix is a 80 degree rotation around the x axis and a smaller translation
  secondMatrix = MakeMatrix ( 80 , 10 );

  trackingAndTimeStamps.Insert (first, firstMatrix);
  trackingAndTimeStamps.Insert (second, secondMatrix);

  cv::Matx44d interpolatedMatrix;
  bool inBounds;
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(10,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, firstMatrix) && ( timingError == 0 ) && inBounds ) , 
      "InterpolateMatrix(10): No interpolation needed at start");
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(20,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, secondMatrix) && ( timingError == 0 ) && inBounds ) , 
      "InterpolateMatrix(20): No interpolation needed at end");
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(15,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, MakeMatrix(45,30)) && ( timingError == 5 ) && inBounds ) , 
      "InterpolateMatrix(15): Halfway : " << timingError );

  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(12,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, MakeMatrix(24,42)) && ( timingError == 2 ) && inBounds ) , 
      "InterpolateMatrix(12): Less than halfway : " << timingError );

  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(16,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, MakeMatrix(52,26)) && ( timingError == 4 ) && inBounds ) , 
      "InterpolateMatrix(16): More than halfway : " << timingError );
  
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(16,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, MakeMatrix(52,26)) && ( timingError == 4 ) && inBounds ) , 
      "InterpolateMatrix(16): More than halfway : " << timingError );
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(22,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, secondMatrix) && ( timingError == 2 ) && (!inBounds) ) , 
      "InterpolateMatrix(22): Past end : " << timingError );
  
  interpolatedMatrix = trackingAndTimeStamps.InterpolateMatrix(2,timingError, inBounds);
  MITK_TEST_CONDITION ( 
      ( CompareMatrices(interpolatedMatrix, firstMatrix) && ( timingError == 8 ) && (!inBounds) ) , 
      "InterpolateMatrix(2): before start : " << timingError );
  MITK_TEST_END();
}


