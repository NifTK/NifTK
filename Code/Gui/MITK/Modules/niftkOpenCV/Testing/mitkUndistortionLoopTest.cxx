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
#include <boost/math/special_functions/fpclassify.hpp>

/**
 * Test for stereo projection and undistortion. Start with a realistic set
 * of 3D points defined relative to the left lens. Project them to screen space, 
 * using project3D including distortion. then undistort. 
 * Then project them to screen space using project 3D without distortion. Compare.
 */


int mitkUndistortionLoopTest ( int argc, char * argv[] )
{

  std::string calibrationDirectory = "";
  double featureDepth = 50;
  bool cropNonVisiblePoints = true; 
  double screenWidth = 1980;
  double screenHeight = 540;
  double cropValue = std::numeric_limits<double>::quiet_NaN();

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
      


  MITK_TEST_BEGIN("mitkUndistortionLoopTest");
  cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_64FC1);
  cv::Mat leftCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat leftCameraDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat leftCameraZeroDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat rightCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightCameraDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat rightCameraZeroDistortion = cv::Mat(1,4,CV_64FC1);
  cv::Mat rightToLeftRotationMatrix = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVector = cv::Mat(3,1,CV_64FC1);
  cv::Mat leftCameraToTracker = cv::Mat(4,4,CV_64FC1);
  
  mitk::LoadStereoCameraParametersFromDirectory (calibrationDirectory,
     &leftCameraIntrinsic,&leftCameraDistortion,&rightCameraIntrinsic,
     &rightCameraDistortion,&rightToLeftRotationMatrix,
     &rightToLeftTranslationVector,&leftCameraToTracker);

  for ( int i = 0 ; i < leftCameraZeroDistortion.rows ; i ++ ) 
  {
     for ( int j = 0 ; j < leftCameraZeroDistortion.cols ; j ++ ) 
     {
        leftCameraZeroDistortion.at<double>(i ,j) = 0.0;
        rightCameraZeroDistortion.at<double>(i ,j) = 0.0;
     }
  }

  CvMat* outputLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* output2DPointsLeft = NULL ;
  CvMat* output2DPointsRight = NULL;
  
  CvMat* outputTrimmedLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputTrimmedLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* outputTrimmed2DPointsLeft = NULL ;
  CvMat* outputTrimmed2DPointsRight = NULL;

  CvMat* outputLeftCameraZeroDistortionWorldPointsIn3D = NULL;
  CvMat* outputLeftCameraZeroDistortionWorldNormalsIn3D = NULL ;
  CvMat* output2DZeroDistortionPointsLeft = NULL ;
  CvMat* output2DZeroDistortionPointsRight = NULL;
  
  CvMat* outputTrimmedLeftCameraZeroDistortionWorldPointsIn3D = NULL;
  CvMat* outputTrimmedLeftCameraZeroDistortionWorldNormalsIn3D = NULL ;
  CvMat* outputTrimmed2DZeroDistortionPointsLeft = NULL ;
  CvMat* outputTrimmed2DZeroDistortionPointsRight = NULL;


  int numberOfPoints = 2601;
  cv::Mat leftCameraWorldPoints = cv::Mat (numberOfPoints,3,CV_64FC1);
  cv::Mat leftCameraWorldNormals = cv::Mat (numberOfPoints,3,CV_64FC1);
  
  cv::Mat leftScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  cv::Mat rightScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  
  cv::Mat leftTrimmedScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  cv::Mat rightTrimmedScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  
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

  mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraWorldPoints,leftCameraWorldNormals,
      leftCameraPositionToFocalPointUnitVector,
      leftCameraIntrinsic,leftCameraDistortion,
      rightCameraIntrinsic,rightCameraDistortion,
      rightToLeftRotationMatrix,rightToLeftTranslationVector,
      outputLeftCameraWorldPointsIn3D,
      outputLeftCameraWorldNormalsIn3D,
      output2DPointsLeft,
      output2DPointsRight);

  mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraWorldPoints,leftCameraWorldNormals,
      leftCameraPositionToFocalPointUnitVector,
      leftCameraIntrinsic,leftCameraZeroDistortion,
      rightCameraIntrinsic,rightCameraZeroDistortion,
      rightToLeftRotationMatrix,rightToLeftTranslationVector,
      outputLeftCameraZeroDistortionWorldPointsIn3D,
      outputLeftCameraZeroDistortionWorldNormalsIn3D,
      output2DZeroDistortionPointsLeft,
      output2DZeroDistortionPointsRight);

  bool cropUndistortedPointsToScreen = true;
  double cropValueInf = std::numeric_limits<double>::infinity();
  mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraWorldPoints,leftCameraWorldNormals,
      leftCameraPositionToFocalPointUnitVector,
      leftCameraIntrinsic,leftCameraDistortion,
      rightCameraIntrinsic,rightCameraDistortion,
      rightToLeftRotationMatrix,rightToLeftTranslationVector,
      outputTrimmedLeftCameraWorldPointsIn3D,
      outputTrimmedLeftCameraWorldNormalsIn3D,
      outputTrimmed2DPointsLeft,
      outputTrimmed2DPointsRight,
      cropUndistortedPointsToScreen,
      0,screenWidth,
      0,screenHeight,cropValueInf
      );
  
   mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraWorldPoints,leftCameraWorldNormals,
      leftCameraPositionToFocalPointUnitVector,
      leftCameraIntrinsic,leftCameraZeroDistortion,
      rightCameraIntrinsic,rightCameraZeroDistortion,
      rightToLeftRotationMatrix,rightToLeftTranslationVector,
      outputTrimmedLeftCameraZeroDistortionWorldPointsIn3D,
      outputTrimmedLeftCameraZeroDistortionWorldNormalsIn3D,
      outputTrimmed2DZeroDistortionPointsLeft,
      outputTrimmed2DZeroDistortionPointsRight,
      cropUndistortedPointsToScreen,
      0,screenWidth,
      0,screenHeight,cropValueInf
      );
 
  mitk::UndistortPoints(output2DPointsLeft, 
      leftCameraIntrinsic,leftCameraDistortion,
      leftScreenPoints);

  mitk::UndistortPoints(output2DPointsRight, 
      rightCameraIntrinsic,rightCameraDistortion,
      rightScreenPoints);

  mitk::UndistortPoints(outputTrimmed2DPointsLeft, 
      leftCameraIntrinsic,leftCameraDistortion,
      leftTrimmedScreenPoints,
      cropNonVisiblePoints, 
      0.0 , screenWidth, 0.0, screenHeight, cropValue);

  mitk::UndistortPoints(outputTrimmed2DPointsRight, 
      rightCameraIntrinsic,rightCameraDistortion,
      rightTrimmedScreenPoints,
      cropNonVisiblePoints, 
      0.0 , screenWidth, 0.0, screenHeight, cropValue);

//now just compare *ScreenPoints with output2DZeroDistortion*

  double xErrorMeanLeft = 0.0;
  double yErrorMeanLeft = 0.0;
  double xErrorMeanTrimmedLeft = 0.0;
  double yErrorMeanTrimmedLeft = 0.0;
  double errorRMSLeft = 0.0;
  double errorRMSTrimmedLeft = 0.0;
 
  double xErrorMeanRight = 0.0;
  double yErrorMeanRight = 0.0;
  double xErrorMeanTrimmedRight = 0.0;
  double yErrorMeanTrimmedRight = 0.0;
  double errorRMSRight = 0.0;
  double errorRMSTrimmedRight = 0.0;
  
  int goodPoints = 0;
  int goodPointsTrimmed = 0;

  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    double XErrorLeft = CV_MAT_ELEM(*output2DZeroDistortionPointsLeft,double,i,0) - leftScreenPoints.at<double>(i,0);
    double YErrorLeft = CV_MAT_ELEM(*output2DZeroDistortionPointsLeft,double,i,1) - leftScreenPoints.at<double>(i,1);
    double XErrorRight = CV_MAT_ELEM(*output2DZeroDistortionPointsRight,double,i,0) - rightScreenPoints.at<double>(i,0);
    double YErrorRight = CV_MAT_ELEM(*output2DZeroDistortionPointsRight,double,i,1) - rightScreenPoints.at<double>(i,1);
    double errorLeft = XErrorLeft * XErrorLeft + YErrorLeft * YErrorLeft;
    double errorRight = XErrorRight * XErrorRight + YErrorRight * YErrorRight;

    if ( !  ( boost::math::isnan(XErrorRight) || boost::math::isnan(XErrorLeft))  )
    {
      goodPoints ++;
      xErrorMeanLeft += XErrorLeft;
      xErrorMeanRight += XErrorRight;
      yErrorMeanLeft += YErrorLeft;
      yErrorMeanRight += YErrorRight;
      errorRMSLeft += errorLeft;
      errorRMSRight += errorRight;
    
      if ( goodPoints % 500 == 1 )
      {
        MITK_INFO << i << " :Left[ " <<  CV_MAT_ELEM(*output2DPointsLeft,double,i,0) << " ] " << 
          CV_MAT_ELEM(*output2DZeroDistortionPointsLeft,double,i,0) << " - " << leftScreenPoints.at<double>(i,0) << " = " <<  XErrorLeft;
      
        MITK_INFO << i << " :Right[ " <<  CV_MAT_ELEM(*output2DPointsRight,double,i,0) << " ] " << 
          CV_MAT_ELEM(*output2DZeroDistortionPointsRight,double,i,0) << " - " << rightScreenPoints.at<double>(i,0) << " = " <<  XErrorRight;
      }
    


    }
   
    double XErrorLeftTrimmed = CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsLeft,double,i,0) - leftTrimmedScreenPoints.at<double>(i,0);
    double YErrorLeftTrimmed = CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsLeft,double,i,1) - leftTrimmedScreenPoints.at<double>(i,1);
    double XErrorRightTrimmed = CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsRight,double,i,0) - rightTrimmedScreenPoints.at<double>(i,0);
    double YErrorRightTrimmed = CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsRight,double,i,1) - rightTrimmedScreenPoints.at<double>(i,1);
    double errorLeftTrimmed = XErrorLeftTrimmed * XErrorLeftTrimmed + YErrorLeftTrimmed * YErrorLeftTrimmed;
    double errorRightTrimmed = XErrorRightTrimmed * XErrorRightTrimmed + YErrorRightTrimmed * YErrorRightTrimmed;

    if ( !  ( boost::math::isnan(XErrorRightTrimmed) || boost::math::isnan(XErrorLeftTrimmed))  )
    {
      goodPointsTrimmed ++;
      xErrorMeanTrimmedLeft += XErrorLeftTrimmed;
      xErrorMeanTrimmedRight += XErrorRightTrimmed;
      yErrorMeanTrimmedLeft += YErrorLeftTrimmed;
      yErrorMeanTrimmedRight += YErrorRightTrimmed;
      errorRMSTrimmedLeft += errorLeftTrimmed;
      errorRMSTrimmedRight += errorRightTrimmed;
      if ( goodPointsTrimmed % 250 == 1 )
      {
        MITK_INFO << i << " :LeftTrimmed[ " <<  CV_MAT_ELEM(*outputTrimmed2DPointsLeft,double,i,0) << " ] " << 
        CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsLeft,double,i,0) << " - " << leftTrimmedScreenPoints.at<double>(i,0) << " = " <<  XErrorLeftTrimmed;
      
        MITK_INFO << i << " :RightTrimmed[ " <<  CV_MAT_ELEM(*outputTrimmed2DPointsRight,double,i,0) << " ] " << 
        CV_MAT_ELEM(*outputTrimmed2DZeroDistortionPointsRight,double,i,0) << " - " << rightTrimmedScreenPoints.at<double>(i,0) << " = " <<  XErrorRightTrimmed;
      }

    }
   
  }

  xErrorMeanLeft /= goodPoints;
  xErrorMeanRight /= goodPoints;
  yErrorMeanLeft /= goodPoints;
  yErrorMeanRight /= goodPoints;
  errorRMSLeft = sqrt(errorRMSLeft/goodPoints);
  errorRMSRight = sqrt(errorRMSRight/goodPoints);
  
  xErrorMeanTrimmedLeft /= goodPointsTrimmed;
  xErrorMeanTrimmedRight /= goodPointsTrimmed;
  yErrorMeanTrimmedLeft /= goodPointsTrimmed;
  yErrorMeanTrimmedRight /= goodPointsTrimmed;
  errorRMSTrimmedLeft = sqrt(errorRMSTrimmedLeft/goodPointsTrimmed);
  errorRMSTrimmedRight = sqrt(errorRMSTrimmedRight/goodPointsTrimmed);
 
  MITK_INFO << "There are " << goodPoints << " untrimmed points";
  MITK_INFO << "Mean Left x error = " << xErrorMeanLeft; 
  MITK_INFO << "Mean Left y error = " << yErrorMeanLeft; 
  MITK_INFO << "RMS Left y error = " << errorRMSLeft; 
  
  MITK_INFO << "Mean Right x error = " << xErrorMeanRight; 
  MITK_INFO << "Mean Right y error = " << yErrorMeanRight; 
  MITK_INFO << "RMS Right y error = " << errorRMSRight; 

  MITK_INFO << "There are " << goodPointsTrimmed << " trimmed points";
  MITK_INFO << "Trimmed Mean Left x error = " << xErrorMeanTrimmedLeft; 
  MITK_INFO << "Trimmed Mean Left y error = " << yErrorMeanTrimmedLeft; 
  MITK_INFO << "Trimmed RMS Left y error = " << errorRMSTrimmedLeft; 
  
  MITK_INFO << "Trimmed Mean Right x error = " << xErrorMeanTrimmedRight; 
  MITK_INFO << "Trimmed Mean Right y error = " << yErrorMeanTrimmedRight; 
  MITK_INFO << "Trimmed RMS Right y error = " << errorRMSTrimmedRight; 





  MITK_TEST_CONDITION (fabs(xErrorMeanLeft) < 1e-3 , "Testing x error mean value for left screen");
  MITK_TEST_CONDITION (fabs(yErrorMeanLeft) < 1e-3 , "Testing y error mean value for left screen");
  MITK_TEST_CONDITION (fabs(errorRMSLeft) < 1e-3 , "Testing RMS error value for left screen");
  
  MITK_TEST_CONDITION (fabs(xErrorMeanRight) < 1e-1 , "Testing x error mean value for right screen");
  MITK_TEST_CONDITION (fabs(yErrorMeanRight) < 1e-1 , "Testing y error mean value for right screen");
  MITK_TEST_CONDITION (fabs(errorRMSRight) < 1e-1 , "Testing RMS error value for right screen");

  MITK_TEST_CONDITION (fabs(xErrorMeanTrimmedLeft) < 1e-3 , "Testing x error mean value for trimmed left screen");
  MITK_TEST_CONDITION (fabs(yErrorMeanTrimmedLeft) < 1e-3 , "Testing y error mean value for trimmed left screen");
  MITK_TEST_CONDITION (fabs(errorRMSTrimmedLeft) < 1e-3 , "Testing RMS error value for trimmed left screen");
  
  MITK_TEST_CONDITION (fabs(xErrorMeanTrimmedRight) < 1e-3 , "Testing x error mean value for trimmed right screen");
  MITK_TEST_CONDITION (fabs(yErrorMeanTrimmedRight) < 1e-3 , "Testing y error mean value for trimmed right screen");
  MITK_TEST_CONDITION (fabs(errorRMSTrimmedRight) < 1e-2 , "Testing RMS error value for trimmed right screen");



  MITK_TEST_END();
}
