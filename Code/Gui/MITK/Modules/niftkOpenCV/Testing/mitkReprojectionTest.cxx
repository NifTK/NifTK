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

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkCameraCalibrationFacade.h>


/**
 * Test for stereo trianglulation and projection. Start with a realistic set
 * of 3D points defined relative to the left lens. Project them to screen space, 
 * then Triangulate them back to world space
 * pair, triangulate them to lens space, then project them back to onscreen 
 * coordinates, they should match
 */


int mitkReprojectionTest ( int argc, char * argv[] )
{

  cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_32FC1);
  cv::Mat leftCameraIntrinsic = cv::Mat(3,3,CV_32FC1);
  cv::Mat leftCameraDistortion = cv::Mat(5,1,CV_32FC1);
  cv::Mat rightCameraIntrinsic = cv::Mat(3,3,CV_32FC1);
  cv::Mat rightCameraDistortion = cv::Mat(5,1,CV_32FC1);
  cv::Mat rightToLeftRotationMatrix = cv::Mat(3,3,CV_32FC1);
  cv::Mat rightToLeftTranslationVector = cv::Mat(3,1,CV_32FC1);
  cv::Mat leftCameraToTracker = cv::Mat(4,4,CV_32FC1);
  
  mitk::LoadStereoCameraParametersFromDirectory (argv[1],
     &leftCameraIntrinsic,&leftCameraDistortion,&rightCameraIntrinsic,
     &rightCameraDistortion,&rightToLeftRotationMatrix,
     &rightToLeftTranslationVector,&leftCameraToTracker);

  CvMat* outputLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* output2DPointsLeft = NULL ;
  CvMat* output2DPointsRight = NULL;
  
  cv::Mat leftCameraWorldPoints = cv::Mat (25,3,CV_32FC1);
  cv::Mat leftCameraWorldNormals = cv::Mat (25,3,CV_32FC1);
  
  for ( int row = 0 ; row < 5 ; row ++ ) 
  {
    for ( int col = 0 ; col < 5 ; col ++ )
    {
      leftCameraWorldPoints.at<float>(row * 5 + col, 0) = -100 + (col * 50);
      leftCameraWorldPoints.at<float>(row * 5 + col, 1) = -100 + (row * 50);
      leftCameraWorldPoints.at<float>(row * 5 + col, 2) = 150 + (row + col) * 1.0;
      leftCameraWorldNormals.at<float>(row*5 + col, 0 ) = 0;
      leftCameraWorldNormals.at<float>(row*5 + col, 1 ) = 0;
      leftCameraWorldNormals.at<float>(row*5 + col, 2 ) = -1.0;
    }
  }
  std::vector<int> Points = mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraWorldPoints,leftCameraWorldNormals,
      leftCameraPositionToFocalPointUnitVector,
      leftCameraIntrinsic,leftCameraDistortion,
      rightCameraIntrinsic,rightCameraDistortion,
      rightToLeftRotationMatrix,rightToLeftTranslationVector,
      outputLeftCameraWorldPointsIn3D,
      outputLeftCameraWorldNormalsIn3D,
      output2DPointsLeft,
      output2DPointsRight);

  for ( int row = 0 ; row < 5 ; row ++ ) 
  {
    for ( int col = 0 ; col < 5 ; col ++ )
    {
      MITK_INFO << "[" << leftCameraWorldPoints.at<float>(row*5 + col,0) << "," 
        << leftCameraWorldPoints.at<float>(row*5 + col,1) << ","
        << leftCameraWorldPoints.at<float>(row*5 + col,2) << "] => ("  
        << CV_MAT_ELEM (*output2DPointsLeft ,float, row*5 + col,0) << "," 
        << CV_MAT_ELEM (*output2DPointsLeft, float,row*5 + col,1) << ") (" 
        << CV_MAT_ELEM (*output2DPointsRight,float,row*5 + col,0) << "," 
        << CV_MAT_ELEM (*output2DPointsRight,float,row*5 + col,1) << ") => "; 
    }
  }

  return 0;
}
