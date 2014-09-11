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

cv::Matx44d MakeMatrix ( double xrot, double xtrans , double ytrans, double ztrans)
{
  cv::Matx44d matrix;
  return matrix;

}
bool CompareMatrices ( cv::Matx44d mat1 , cv::Matx44d mat2 )
{
  double precision = 1e-6;
  double error = 0;
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
  cv::Matx44d firstMatrix;
  cv::Matx44d secondMatrix;

  //first matrix is a 10 degree rotation around the x axis and a small translation
  //
  trackingAndTimeStamps.Insert (first, firstMatrix);
  trackingAndTimeStamps.Insert (second, secondMatrix);

 
  MITK_TEST_END();
}


