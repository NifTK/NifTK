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

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

/**
 * Test for stereo trianglulation and projection. Start with a realistic set
 * of 3D points defined relative to the left lens. Project them to screen space, 
 * then Triangulate them back to world space
 * pair, triangulate them to lens space, then project them back to onscreen 
 * coordinates, they should match
 */


int mitkReprojectionTest ( int argc, char * argv[] )
{

  std::string calibrationDirectory = "";
  double pixelNoise = 0.0;
  bool quantize = false;
  double featureDepth = 50;
  bool cropNonVisiblePoints = true; 
  double screenWidth = 1980;
  double screenHeight = 540;

  bool ok;
  while ( argc > 1 ) 
  {
    ok = false;
    if (( ok == false ) && (strcmp(argv[1],"--calibration") == 0 ))
    {
      argc--;
      argv++;
      calibrationDirectory = argv[1];
      MITK_INFO << "Loading calibration from " << calibrationDirectory;
      argc--;
      argv++;
      ok=true;
    }
    if (( ok == false ) && (strcmp(argv[1],"--pixelNoise") == 0 )) 
    {
      argc--;
      argv++;
      pixelNoise = atof(argv[1]);
      argc--;
      argv++;
      ok =true;
    }
    if (( ok == false ) && (strcmp(argv[1],"--quantize") == 0 )) 
    {
      argc--;
      argv++;
      quantize=true;
      ok=true;
    }
    if (( ok == false ) && ( strcmp (argv[1],"--featureDepth") == 0 ))
    {
      argc--;
      argv++;
      featureDepth = atof(argv[1]);
      argc--;
      argv++;
      ok=true;
    }
    if ( ok == false )
    {
      MITK_ERROR << "Failed to parse arguments";
      return EXIT_FAILURE;
    }
  }
      


  MITK_TEST_BEGIN("mitkReprojectionTest");
  cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_64FC1);
  cv::Mat leftCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat leftCameraDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat rightCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightCameraDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat rightToLeftRotationMatrix = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVector = cv::Mat(3,1,CV_64FC1);
  cv::Mat leftCameraToTracker = cv::Mat(4,4,CV_64FC1);
  
  mitk::LoadStereoCameraParametersFromDirectory (calibrationDirectory,
     &leftCameraIntrinsic,&leftCameraDistortion,&rightCameraIntrinsic,
     &rightCameraDistortion,&rightToLeftRotationMatrix,
     &rightToLeftTranslationVector,&leftCameraToTracker);

  CvMat* outputLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* output2DPointsLeft = NULL ;
  CvMat* output2DPointsRight = NULL;
  
  int numberOfPoints = 2601;
  cv::Mat leftCameraWorldPoints = cv::Mat (numberOfPoints,3,CV_64FC1);
  cv::Mat leftCameraWorldNormals = cv::Mat (numberOfPoints,3,CV_64FC1);
  
  CvMat* leftCameraTriangulatedWorldPoints_m1 = cvCreateMat (numberOfPoints,3,CV_64FC1);
  cv::Mat leftScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  cv::Mat rightScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  
  for ( int row = 0 ; row < 51 ; row ++ ) 
  {
    for ( int col = 0 ; col < 51 ; col ++ )
    {
      leftCameraWorldPoints.at<double>(row * 51 + col, 0) = -25 + (col);
      leftCameraWorldPoints.at<double>(row * 51 + col, 1) = -25 + (row);
      leftCameraWorldPoints.at<double>(row * 51 + col, 2) = featureDepth;
      leftCameraWorldNormals.at<double>(row*51 + col, 0 ) = 0;
      leftCameraWorldNormals.at<double>(row*51 + col, 1 ) = 0;
      leftCameraWorldNormals.at<double>(row*51 + col, 2 ) = -1.0;
    }
  }
  leftCameraPositionToFocalPointUnitVector.at<double>(0,0)=0;
  leftCameraPositionToFocalPointUnitVector.at<double>(0,1)=0;
  leftCameraPositionToFocalPointUnitVector.at<double>(0,2)=1.0;

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
 
  boost::mt19937 rng;
  boost::normal_distribution<> nd(0.0,pixelNoise);
  boost::variate_generator<boost::mt19937& , boost::normal_distribution<> > var_nor (rng,nd);
  MITK_INFO << "GREP ME " << var_nor() << " " << var_nor() << " " << var_nor();
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    CV_MAT_ELEM (*output2DPointsLeft ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsLeft ,double,i,1) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsRight ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsRight ,double,i,1) += var_nor(); 
  }

  if ( quantize ) 
  {
    for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
    {
      CV_MAT_ELEM (*output2DPointsLeft ,double,i,0) =
        floor ( CV_MAT_ELEM (*output2DPointsLeft ,double,i,0) + 0.5 );
      CV_MAT_ELEM (*output2DPointsLeft ,double,i,1) =
        floor ( CV_MAT_ELEM (*output2DPointsLeft ,double,i,1) + 0.5 );
      CV_MAT_ELEM (*output2DPointsRight ,double,i,0) =
        floor ( CV_MAT_ELEM (*output2DPointsRight ,double,i,0) + 0.5 );
      CV_MAT_ELEM (*output2DPointsRight ,double,i,1) =
        floor ( CV_MAT_ELEM (*output2DPointsRight ,double,i,1) + 0.5 );
    }
  }
  
  mitk::UndistortPoints(output2DPointsLeft, 
      leftCameraIntrinsic,leftCameraDistortion,
      leftScreenPoints);

  mitk::UndistortPoints(output2DPointsRight, 
      rightCameraIntrinsic,rightCameraDistortion,
      rightScreenPoints);

 //check it with the c Wrapper function
  cv::Mat leftCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat leftCameraRotationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat rightCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat rightCameraRotationVector = cv::Mat (3,1,CV_64FC1);
 
  for ( int i = 0 ; i < 3 ; i ++ ) 
  {
    leftCameraTranslationVector.at<double>(i,0) = 0.0;
    leftCameraRotationVector.at<double>(i,0) = 0.0;
  }
  rightCameraTranslationVector = rightToLeftTranslationVector * -1;
  cv::Rodrigues ( rightToLeftRotationMatrix.inv(), rightCameraRotationVector  );
  
  MITK_INFO << leftCameraTranslationVector;
  MITK_INFO << leftCameraRotationVector;
  MITK_INFO << rightCameraTranslationVector;
  MITK_INFO << rightCameraRotationVector;

  CvMat leftScreenPointsMat = leftScreenPoints;// cvCreateMat(numberOfPoints,2,CV_64FC1;
  CvMat rightScreenPointsMat= rightScreenPoints; 
  CvMat leftCameraIntrinsicMat= leftCameraIntrinsic;
  CvMat leftCameraRotationVectorMat= leftCameraRotationVector; 
  CvMat leftCameraTranslationVectorMat= leftCameraTranslationVector;
  CvMat rightCameraIntrinsicMat = rightCameraIntrinsic;
  CvMat rightCameraRotationVectorMat = rightCameraRotationVector;
  CvMat rightCameraTranslationVectorMat= rightCameraTranslationVector;

  mitk::CStyleTriangulatePointPairsUsingSVD(
    leftScreenPointsMat,
    rightScreenPointsMat,
    leftCameraIntrinsicMat,
    leftCameraRotationVectorMat,
    leftCameraTranslationVectorMat,
    rightCameraIntrinsicMat,
    rightCameraRotationVectorMat,
    rightCameraTranslationVectorMat,
    *leftCameraTriangulatedWorldPoints_m1);

  std::vector < std::pair<cv::Point2d, cv::Point2d> > inputUndistortedPoints;
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    std::pair <cv::Point2d, cv::Point2d > pointPair; 
    pointPair.first.x = leftScreenPoints.at<double>(i,0);
    pointPair.first.y = leftScreenPoints.at<double>(i,1);
    pointPair.second.x = rightScreenPoints.at<double>(i,0);
    pointPair.second.y = rightScreenPoints.at<double>(i,1);
    inputUndistortedPoints.push_back(pointPair);
  }
  cv::Mat rightToLeftRotationVector(3,1,CV_64FC1);
  cv::Rodrigues( rightToLeftRotationMatrix, rightToLeftRotationVector);
  std::vector <cv::Point3d> leftCameraTriangulatedWorldPoints_m2 = 
    mitk::TriangulatePointPairsUsingGeometry(
        inputUndistortedPoints, 
        leftCameraIntrinsic,
        rightCameraIntrinsic,
        rightToLeftRotationVector,
        rightToLeftTranslationVector,
        100.0 // don't know tolerance allowable yet.
        );

  MITK_INFO << leftCameraTriangulatedWorldPoints_m2.size();

    
  for ( int row = 0 ; row < 51 ; row += 25 ) 
  {
    for ( int col = 0 ; col < 51 ; col += 25 )
    {
      MITK_INFO << "(" << row << "," << col <<  ") [" << leftCameraWorldPoints.at<double>(row*51 + col,0) << "," 
        << leftCameraWorldPoints.at<double>(row*51 + col,1) << ","
        << leftCameraWorldPoints.at<double>(row*51 + col,2) << "] => ("  
        << CV_MAT_ELEM (*output2DPointsLeft ,double, row*51 + col,0) << "," 
        << CV_MAT_ELEM (*output2DPointsLeft, double,row*51 + col,1) << ") (" 
        << CV_MAT_ELEM (*output2DPointsRight,double,row*51 + col,0) << "," 
        << CV_MAT_ELEM (*output2DPointsRight,double,row*51 + col,1) << ") => "
        << " [" 
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, row* 51 + col, 0) << ","
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, row* 51 + col, 1) << ","
        << CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, row* 51 + col, 2) << "] " 
        << leftCameraTriangulatedWorldPoints_m2[row*51 + col]; 
    }
  }

  double xErrorMean_m1 = 0.0;
  double yErrorMean_m1 = 0.0;
  double zErrorMean_m1 = 0.0;
  double xErrorMean_m2 = 0.0;
  double yErrorMean_m2 = 0.0;
  double zErrorMean_m2 = 0.0;
  double errorRMS_m1 = 0.0;
  double errorRMS_m2 = 0.0;
  
  int goodPoints = 0;
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {

    double xError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 0) - 
      leftCameraWorldPoints.at<double>(i,0);
    double yError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 1) - 
      leftCameraWorldPoints.at<double>(i,1);
    double zError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 2) - 
      leftCameraWorldPoints.at<double>(i,2);
    double xError_m2 = leftCameraTriangulatedWorldPoints_m2[i].x -  
      leftCameraWorldPoints.at<double>(i,0);
    double yError_m2 = leftCameraTriangulatedWorldPoints_m2[i].y -  
      leftCameraWorldPoints.at<double>(i,1);
    double zError_m2 = leftCameraTriangulatedWorldPoints_m2[i].z -  
      leftCameraWorldPoints.at<double>(i,2);
    
    double error_m1 = (xError_m1 * xError_m1 + yError_m1 * yError_m1 + zError_m1 * zError_m1);
    double error_m2 = (xError_m2 * xError_m2 + yError_m2 * yError_m2 + zError_m2 * zError_m2);
    
    if ( ! cropNonVisiblePoints || ! (
        ( leftScreenPoints.at<double>(i,0) < 0.0) ||
           ( leftScreenPoints.at<double>(i,0) > screenWidth )  || 
           ( leftScreenPoints.at<double>(i,1) < 0.0 ) || 
           ( leftScreenPoints.at<double>(i,1) > screenHeight ) || 
           ( rightScreenPoints.at<double>(i,0) < 0.0) ||
           ( rightScreenPoints.at<double>(i,0) > screenWidth )  || 
           ( rightScreenPoints.at<double>(i,1) < 0.0 ) || 
           ( rightScreenPoints.at<double>(i,1) > screenHeight ) ) )
    {
      xErrorMean_m1 += xError_m1;
      yErrorMean_m1 += yError_m1;
      zErrorMean_m1 += zError_m1;
      xErrorMean_m2 += xError_m2;
      yErrorMean_m2 += yError_m2;
      zErrorMean_m2 += zError_m2;
      errorRMS_m1 += error_m1;
      errorRMS_m2 += error_m2;
      goodPoints++;
    }
  }
  MITK_INFO << "Dumped " << numberOfPoints - goodPoints << " off screen points";
  xErrorMean_m1 /= goodPoints;
  yErrorMean_m1 /= goodPoints;
  zErrorMean_m1 /= goodPoints;
  xErrorMean_m2 /= goodPoints;
  yErrorMean_m2 /= goodPoints;
  zErrorMean_m2 /= goodPoints;
  errorRMS_m1 = sqrt(errorRMS_m1/goodPoints);
  errorRMS_m2 = sqrt(errorRMS_m2/goodPoints);
  
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
  MITK_TEST_CONDITION (errorRMS_m1 < 1e-3 , "Testing RMS error value for c method");
  MITK_TEST_CONDITION (fabs(xErrorMean_m2) < 0.5 , "Testing x error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(yErrorMean_m2) < 0.5 , "Testing y error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(zErrorMean_m2) < 0.5 , "Testing z error mean value for c++ method");
  MITK_TEST_CONDITION (errorRMS_m2 < 2.0 , "Testing RMS error value for c++ method");

  MITK_TEST_END();
}
