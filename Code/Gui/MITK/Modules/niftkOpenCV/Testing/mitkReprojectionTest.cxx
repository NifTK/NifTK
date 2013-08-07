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
  MITK_TEST_BEGIN("mitkReprojectionTest");
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
  
  int numberOfPoints = 25;
  cv::Mat leftCameraWorldPoints = cv::Mat (numberOfPoints,3,CV_32FC1);
  cv::Mat leftCameraWorldNormals = cv::Mat (numberOfPoints,3,CV_32FC1);
  
  CvMat* leftCameraTriangulatedWorldPoints_m1 = cvCreateMat (numberOfPoints,3,CV_32FC1);
  cv::Mat leftScreenPoints = cv::Mat (numberOfPoints,2,CV_32FC1);
  cv::Mat rightScreenPoints = cv::Mat (numberOfPoints,2,CV_32FC1);
  
  for ( int row = 0 ; row < 5 ; row ++ ) 
  {
    for ( int col = 0 ; col < 5 ; col ++ )
    {
      leftCameraWorldPoints.at<float>(row * 5 + col, 0) = -50 + (col * 25);
      leftCameraWorldPoints.at<float>(row * 5 + col, 1) = -30 + (row * 15);
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

  mitk::UndistortPoints(output2DPointsLeft, 
      leftCameraIntrinsic,leftCameraDistortion,
      leftScreenPoints);

  mitk::UndistortPoints(output2DPointsRight, 
      rightCameraIntrinsic,rightCameraDistortion,
      rightScreenPoints);

  //check it with the c Wrapper function
  cv::Mat leftCameraTranslationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat leftCameraRotationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat rightCameraTranslationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat rightCameraRotationVector = cv::Mat (3,1,CV_32FC1);
  
  rightCameraTranslationVector = rightToLeftTranslationVector * -1;
  cv::Rodrigues ( rightToLeftRotationMatrix.inv(), rightCameraRotationVector  );
  
  MITK_INFO << leftCameraTranslationVector;
  MITK_INFO << leftCameraRotationVector;
  MITK_INFO << rightCameraTranslationVector;
  MITK_INFO << rightCameraRotationVector;

  CvMat leftScreenPointsMat = leftScreenPoints;// cvCreateMat(numberOfPoints,2,CV_32FC1;
  CvMat rightScreenPointsMat= rightScreenPoints; 
  CvMat leftCameraIntrinsicMat= leftCameraIntrinsic;
  CvMat leftCameraRotationVectorMat= leftCameraRotationVector; 
  CvMat leftCameraTranslationVectorMat= leftCameraTranslationVector;
  CvMat rightCameraIntrinsicMat = rightCameraIntrinsic;
  CvMat rightCameraRotationVectorMat = rightCameraRotationVector;
  CvMat rightCameraTranslationVectorMat= rightCameraTranslationVector;

  mitk::TriangulatePointPairs(
    leftScreenPointsMat,
    rightScreenPointsMat,
    leftCameraIntrinsicMat,
    leftCameraRotationVectorMat,
    leftCameraTranslationVectorMat,
    rightCameraIntrinsicMat,
    rightCameraRotationVectorMat,
    rightCameraTranslationVectorMat,
    *leftCameraTriangulatedWorldPoints_m1);

  std::vector < std::pair<cv::Point2f, cv::Point2f> > inputUndistortedPoints;
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    std::pair <cv::Point2f, cv::Point2f > pointPair; 
    pointPair.first.x = leftScreenPoints.at<float>(i,0);
    pointPair.first.y = leftScreenPoints.at<float>(i,1);
    pointPair.second.x = rightScreenPoints.at<float>(i,0);
    pointPair.second.y = rightScreenPoints.at<float>(i,1);
    inputUndistortedPoints.push_back(pointPair);
  }
  cv::Mat rightToLeftRotationVector(3,1,CV_32FC1);
  cv::Rodrigues( rightToLeftRotationMatrix, rightToLeftRotationVector);
  std::vector <cv::Point3f> leftCameraTriangulatedWorldPoints_m2 = 
    mitk::TriangulatePointPairs(
        inputUndistortedPoints, 
        leftCameraIntrinsic,
        rightCameraIntrinsic,
        rightToLeftRotationVector,
        rightToLeftTranslationVector);

  MITK_INFO << leftCameraTriangulatedWorldPoints_m2.size();

    
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
        << CV_MAT_ELEM (*output2DPointsRight,float,row*5 + col,1) << ") => "
        << " [" 
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, row* 5 + col, 0) << ","
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, row* 5 + col, 1) << ","
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, row* 5 + col, 2) << "] " 
        << leftCameraTriangulatedWorldPoints_m2[row*5 + col]; 
    }
  }

  float xErrorMean_m1 = 0.0;
  float yErrorMean_m1 = 0.0;
  float zErrorMean_m1 = 0.0;
  float xErrorMean_m2 = 0.0;
  float yErrorMean_m2 = 0.0;
  float zErrorMean_m2 = 0.0;
  float errorRMS_m1 = 0.0;
  float errorRMS_m2 = 0.0;

  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    float xError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, i, 0) - 
      leftCameraWorldPoints.at<float>(i,0);
    float yError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, i, 1) - 
      leftCameraWorldPoints.at<float>(i,1);
    float zError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,float, i, 2) - 
      leftCameraWorldPoints.at<float>(i,2);
    float xError_m2 = leftCameraTriangulatedWorldPoints_m2[i].x -  
      leftCameraWorldPoints.at<float>(i,0);
    float yError_m2 = leftCameraTriangulatedWorldPoints_m2[i].y -  
      leftCameraWorldPoints.at<float>(i,1);
    float zError_m2 = leftCameraTriangulatedWorldPoints_m2[i].z -  
      leftCameraWorldPoints.at<float>(i,2);
    float error_m1 = (xError_m1 * xError_m1 + yError_m1 * yError_m1 + zError_m1 * zError_m1);
    float error_m2 = (xError_m2 * xError_m2 + yError_m2 * yError_m2 + zError_m2 * zError_m2);
    
    xErrorMean_m1 += xError_m1;
    yErrorMean_m1 += yError_m1;
    zErrorMean_m1 += zError_m1;
    xErrorMean_m2 += xError_m2;
    yErrorMean_m2 += yError_m2;
    zErrorMean_m2 += zError_m2;
    errorRMS_m1 += error_m1;
    errorRMS_m2 += error_m2;
  }
  xErrorMean_m1 /= numberOfPoints;
  yErrorMean_m1 /= numberOfPoints;
  zErrorMean_m1 /= numberOfPoints;
  xErrorMean_m2 /= numberOfPoints;
  yErrorMean_m2 /= numberOfPoints;
  zErrorMean_m2 /= numberOfPoints;
  errorRMS_m1 = sqrt(errorRMS_m1/numberOfPoints);
  errorRMS_m2 = sqrt(errorRMS_m2/numberOfPoints);
  
  MITK_INFO << "Mean x error c wrapper = " <<  xErrorMean_m1; 
  MITK_INFO << "Mean y error c wrapper = " <<  yErrorMean_m1; 
  MITK_INFO << "Mean z error c wrapper = " <<  zErrorMean_m1; 
  MITK_INFO << "RMS error c+wrapper = " <<  errorRMS_m1; 
  MITK_INFO << "Mean x error c++ wrapper = " <<  xErrorMean_m2; 
  MITK_INFO << "Mean y error c++ wrapper = " <<  yErrorMean_m2; 
  MITK_INFO << "Mean z error c++ wrapper = " <<  zErrorMean_m2; 
  MITK_INFO << "RMS error c++ wrapper = " <<  errorRMS_m2; 
  MITK_TEST_CONDITION (fabs(xErrorMean_m1) < 1e-3 , "Testing x error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(yErrorMean_m1) < 1e-3 , "Testing y error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(zErrorMean_m1) < 1e-3 , "Testing z error mean value for c wrapper method");
  MITK_TEST_CONDITION (errorRMS_m1 < 1.0 , "Testing RMS error value for c method");
  MITK_TEST_CONDITION (fabs(xErrorMean_m2) < 1e-3 , "Testing x error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(yErrorMean_m2) < 1e-3 , "Testing y error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(zErrorMean_m2) < 1e-3 , "Testing z error mean value for c++ method");
  MITK_TEST_CONDITION (errorRMS_m2 < 1.0 , "Testing RMS error value for c++ method");

  MITK_TEST_END();
}
