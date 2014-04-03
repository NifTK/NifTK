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

  // Section to output reprojected image size
  cv::Point2d topLeft = cv::Point2d (0,0);
  cv::Point2d bottomRight = cv::Point2d (screenWidth, screenHeight);
  cv::Point2d undistortedTopLeft;
  cv::Point2d undistortedBottomRight;
  mitk::UndistortPoint ( topLeft, leftCameraIntrinsic, leftCameraDistortion, undistortedTopLeft);
  mitk::UndistortPoint ( bottomRight, leftCameraIntrinsic, leftCameraDistortion, undistortedBottomRight);

  cv::Point3d reProjectedTopLeft = mitk::ReProjectPoint (undistortedTopLeft, leftCameraIntrinsic);
  cv::Point3d reProjectedBottomRight = mitk::ReProjectPoint (undistortedBottomRight, leftCameraIntrinsic);

  reProjectedTopLeft.x *= featureDepth;
  reProjectedTopLeft.y *= featureDepth;
  reProjectedTopLeft.z *= featureDepth;

  reProjectedBottomRight.x *= featureDepth;
  reProjectedBottomRight.y *= featureDepth;
  reProjectedBottomRight.z *= featureDepth;

  double diagonalSize = sqrt (
       pow ( (reProjectedTopLeft.x - reProjectedBottomRight.x),2.0) +
       pow ( (reProjectedTopLeft.y - reProjectedBottomRight.y),2.0) );

  MITK_INFO << "Top Left = " << reProjectedTopLeft;
  MITK_INFO << "Bottom Right = " << reProjectedBottomRight;
  MITK_INFO << "Length of Diagonal = " << diagonalSize;
  // End of reprojected image size bit
  

  CvMat* outputLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* output2DPointsLeft = NULL ;
  CvMat* output2DPointsRight = NULL;
  
  CvMat* outputTrimmedLeftCameraWorldPointsIn3D = NULL;
  CvMat* outputTrimmedLeftCameraWorldNormalsIn3D = NULL ;
  CvMat* outputTrimmed2DPointsLeft = NULL ;
  CvMat* outputTrimmed2DPointsRight = NULL;

  int numberOfPoints = 2601;
  cv::Mat leftCameraWorldPoints = cv::Mat (numberOfPoints,3,CV_64FC1);
  cv::Mat leftCameraWorldNormals = cv::Mat (numberOfPoints,3,CV_64FC1);
  
  CvMat* leftCameraTriangulatedWorldPoints_m1 = cvCreateMat (numberOfPoints,3,CV_64FC1);
  cv::Mat leftScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  cv::Mat rightScreenPoints = cv::Mat (numberOfPoints,2,CV_64FC1);
  
  CvMat* leftCameraTriangulatedWorldPoints_trimmed_m1 = cvCreateMat (numberOfPoints,3,CV_64FC1);
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

  bool cropUndistortedPointsToScreen = true;
  double cropValueInf = std::numeric_limits<double>::infinity();
  std::vector<int> trimmedPoints = mitk::ProjectVisible3DWorldPointsToStereo2D
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
 

  boost::mt19937 rng;
  boost::normal_distribution<> nd(0.0,pixelNoise);
  boost::variate_generator<boost::mt19937& , boost::normal_distribution<> > var_nor (rng,nd);
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    CV_MAT_ELEM (*output2DPointsLeft ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsLeft ,double,i,1) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsRight ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*output2DPointsRight ,double,i,1) += var_nor(); 

    CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,1) += var_nor(); 
    CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,0) += var_nor(); 
    CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,1) += var_nor(); 
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

      CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,0) =
        floor ( CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,0) + 0.5 );
      CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,1) =
        floor ( CV_MAT_ELEM (*outputTrimmed2DPointsLeft ,double,i,1) + 0.5 );
      CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,0) =
        floor ( CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,0) + 0.5 );
      CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,1) =
        floor ( CV_MAT_ELEM (*outputTrimmed2DPointsRight ,double,i,1) + 0.5 );
    }
  }
  
 
  mitk::UndistortPoints(output2DPointsLeft, 
      leftCameraIntrinsic,leftCameraDistortion,
      leftScreenPoints,
      cropNonVisiblePoints, 
      0.0 , screenWidth, 0.0, screenHeight, cropValue);

  mitk::UndistortPoints(output2DPointsRight, 
      rightCameraIntrinsic,rightCameraDistortion,
      rightScreenPoints,
      cropNonVisiblePoints, 
      0.0 , screenWidth, 0.0, screenHeight, cropValue);

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


 //check it with the c Wrapper function
  cv::Mat leftCameraTranslationVector = cv::Mat (1,3,CV_64FC1);
  cv::Mat leftCameraRotationVector = cv::Mat (1,3,CV_64FC1);
  cv::Mat rightCameraTranslationVector = cv::Mat (1,3,CV_64FC1);
  cv::Mat rightCameraRotationVector = cv::Mat (1,3,CV_64FC1);
 
  for ( int i = 0 ; i < 3 ; i ++ ) 
  {
    leftCameraTranslationVector.at<double>(0,i) = 0.0;
    leftCameraRotationVector.at<double>(0,i) = 0.0;
  }
  rightCameraTranslationVector = rightToLeftTranslationVector * -1;
  cv::Rodrigues ( rightToLeftRotationMatrix.inv(), rightCameraRotationVector  );
  
  MITK_DEBUG << leftCameraTranslationVector;
  MITK_DEBUG << leftCameraRotationVector;
  MITK_DEBUG << rightCameraTranslationVector;
  MITK_DEBUG << rightCameraRotationVector;

  CvMat leftScreenPointsMat = leftScreenPoints;// cvCreateMat(numberOfPoints,2,CV_64FC1;
  CvMat rightScreenPointsMat= rightScreenPoints; 
  CvMat leftTrimmedScreenPointsMat = leftTrimmedScreenPoints;// cvCreateMat(numberOfPoints,2,CV_64FC1;
  CvMat rightTrimmedScreenPointsMat= rightTrimmedScreenPoints; 
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

  mitk::CStyleTriangulatePointPairsUsingSVD(
    leftTrimmedScreenPointsMat,
    rightTrimmedScreenPointsMat,
    leftCameraIntrinsicMat,
    leftCameraRotationVectorMat,
    leftCameraTranslationVectorMat,
    rightCameraIntrinsicMat,
    rightCameraRotationVectorMat,
    rightCameraTranslationVectorMat,
    *leftCameraTriangulatedWorldPoints_trimmed_m1);

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
  std::vector < std::pair<cv::Point2d, cv::Point2d> > inputTrimmedUndistortedPoints;
  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    std::pair <cv::Point2d, cv::Point2d > pointPair; 
    pointPair.first.x = leftTrimmedScreenPoints.at<double>(i,0);
    pointPair.first.y = leftTrimmedScreenPoints.at<double>(i,1);
    pointPair.second.x = rightTrimmedScreenPoints.at<double>(i,0);
    pointPair.second.y = rightTrimmedScreenPoints.at<double>(i,1);
    inputTrimmedUndistortedPoints.push_back(pointPair);
  }
  std::vector <cv::Point3d> leftCameraTriangulatedWorldPoints_m2 = 
    mitk::TriangulatePointPairsUsingGeometry(
        inputUndistortedPoints, 
        leftCameraIntrinsic,
        rightCameraIntrinsic,
        rightToLeftRotationMatrix,
        rightToLeftTranslationVector,
        100.0 // don't know tolerance allowable yet.
        );

   std::vector <cv::Point3d> leftCameraTriangulatedWorldPoints_trimmed_m2 = 
    mitk::TriangulatePointPairsUsingGeometry(
        inputTrimmedUndistortedPoints, 
        leftCameraIntrinsic,
        rightCameraIntrinsic,
        rightToLeftRotationMatrix,
        rightToLeftTranslationVector,
        100.0 // don't know tolerance allowable yet.
        );

 MITK_INFO << "size of triangulated point vector = " <<  leftCameraTriangulatedWorldPoints_m2.size();
 MITK_INFO << "size of trimmed triangulated point vector = " <<  leftCameraTriangulatedWorldPoints_trimmed_m2.size();
  //mitk::TriangulatePointPairsUsingGeometry does not ensure that the indexes in inputUndistortedPoints
  //are preserved in  leftCameraTriangulatedWorldPoints_m2, let's add a NaN to the end and
  //build a look up table
  cv::Point3d nanPoint = cv::Point3d(std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
  leftCameraTriangulatedWorldPoints_m2.push_back(nanPoint);
  leftCameraTriangulatedWorldPoints_trimmed_m2.push_back(nanPoint);

  unsigned int leftCameraTriangulatedWorldPoints_Counter = 0;
  unsigned int leftTrimmedCameraTriangulatedWorldPoints_Counter = 0;
  std::vector < unsigned int > leftCameraTriangulatedWorldPoints_LookUpVector;
  std::vector < unsigned int > leftTrimmedCameraTriangulatedWorldPoints_LookUpVector;
  for ( int i = 0 ; i < numberOfPoints ; i ++ )
  {
    if ( ( ! boost::math::isnan(inputUndistortedPoints[i].first.x) ) &&
          ( ! boost::math::isnan(inputUndistortedPoints[i].second.x) ) )
    {
      leftCameraTriangulatedWorldPoints_LookUpVector.push_back(leftCameraTriangulatedWorldPoints_Counter);
      leftCameraTriangulatedWorldPoints_Counter++;
    }
    else
    {
      leftCameraTriangulatedWorldPoints_LookUpVector.push_back(leftCameraTriangulatedWorldPoints_m2.size()-1);
    }
    if ( ( ! boost::math::isnan(inputTrimmedUndistortedPoints[i].first.x) ) &&
          ( ! boost::math::isnan(inputTrimmedUndistortedPoints[i].second.x) ) )
    {
      leftTrimmedCameraTriangulatedWorldPoints_LookUpVector.push_back(leftTrimmedCameraTriangulatedWorldPoints_Counter);
      leftTrimmedCameraTriangulatedWorldPoints_Counter++;
    }
    else
    {
      leftTrimmedCameraTriangulatedWorldPoints_LookUpVector.push_back(leftCameraTriangulatedWorldPoints_trimmed_m2.size()-1);
    }

  }
          
    
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
        << leftCameraTriangulatedWorldPoints_m2[
        leftCameraTriangulatedWorldPoints_LookUpVector[row*51 + col]]; 
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
  
  int goodPoints_m1 = 0;
  int goodPoints_m2 = 0;

  double xErrorMean_trimmed_m1 = 0.0;
  double yErrorMean_trimmed_m1 = 0.0;
  double zErrorMean_trimmed_m1 = 0.0;
  double xErrorMean_trimmed_m2 = 0.0;
  double yErrorMean_trimmed_m2 = 0.0;
  double zErrorMean_trimmed_m2 = 0.0;
  double errorRMS_trimmed_m1 = 0.0;
  double errorRMS_trimmed_m2 = 0.0;
  
  int goodPoints_trimmed_m1 = 0;
  int goodPoints_trimmed_m2 = 0;

  for ( int i = 0 ; i < numberOfPoints ; i ++ ) 
  {
    double xError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 0) - 
      leftCameraWorldPoints.at<double>(i,0);
    double yError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 1) - 
      leftCameraWorldPoints.at<double>(i,1);
    double zError_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_m1,double, i, 2) - 
      leftCameraWorldPoints.at<double>(i,2);
    double xError_m2 = leftCameraTriangulatedWorldPoints_m2[leftCameraTriangulatedWorldPoints_LookUpVector[i]].x -  
      leftCameraWorldPoints.at<double>(i,0);
    double yError_m2 = leftCameraTriangulatedWorldPoints_m2[leftCameraTriangulatedWorldPoints_LookUpVector[i]].y -  
      leftCameraWorldPoints.at<double>(i,1);
    double zError_m2 = leftCameraTriangulatedWorldPoints_m2[leftCameraTriangulatedWorldPoints_LookUpVector[i]].z -  
      leftCameraWorldPoints.at<double>(i,2);
    
    double error_m1 = (xError_m1 * xError_m1 + yError_m1 * yError_m1 + zError_m1 * zError_m1);
    double error_m2 = (xError_m2 * xError_m2 + yError_m2 * yError_m2 + zError_m2 * zError_m2);
    
    if ( ! boost::math::isnan(error_m1) ) 
    {
      xErrorMean_m1 += xError_m1;
      yErrorMean_m1 += yError_m1;
      zErrorMean_m1 += zError_m1;
      errorRMS_m1 += error_m1;
      goodPoints_m1++;
    }
    if ( ! boost::math::isnan(error_m2 ))
    {
      xErrorMean_m2 += xError_m2;
      yErrorMean_m2 += yError_m2;
      zErrorMean_m2 += zError_m2;
      errorRMS_m2 += error_m2;
      goodPoints_m2++;
    }

    double xError_trimmed_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_trimmed_m1,double, i, 0) - 
      leftCameraWorldPoints.at<double>(i,0);
    double yError_trimmed_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_trimmed_m1,double, i, 1) - 
      leftCameraWorldPoints.at<double>(i,1);
    double zError_trimmed_m1 = CV_MAT_ELEM (*leftCameraTriangulatedWorldPoints_trimmed_m1,double, i, 2) - 
      leftCameraWorldPoints.at<double>(i,2);
    double xError_trimmed_m2 = leftCameraTriangulatedWorldPoints_trimmed_m2[leftTrimmedCameraTriangulatedWorldPoints_LookUpVector[i]].x -  
      leftCameraWorldPoints.at<double>(i,0);
    double yError_trimmed_m2 = leftCameraTriangulatedWorldPoints_trimmed_m2[leftTrimmedCameraTriangulatedWorldPoints_LookUpVector[i]].y -  
      leftCameraWorldPoints.at<double>(i,1);
    double zError_trimmed_m2 = leftCameraTriangulatedWorldPoints_trimmed_m2[leftTrimmedCameraTriangulatedWorldPoints_LookUpVector[i]].z -  
      leftCameraWorldPoints.at<double>(i,2);
    
    double error_trimmed_m1 = (xError_trimmed_m1 * xError_trimmed_m1 + yError_trimmed_m1 * yError_trimmed_m1 + zError_trimmed_m1 * zError_trimmed_m1);
    double error_trimmed_m2 = (xError_trimmed_m2 * xError_trimmed_m2 + yError_trimmed_m2 * yError_trimmed_m2 + zError_trimmed_m2 * zError_trimmed_m2);
    
    if ( ! boost::math::isnan(error_trimmed_m1) ) 
    {
      xErrorMean_trimmed_m1 += xError_trimmed_m1;
      yErrorMean_trimmed_m1 += yError_trimmed_m1;
      zErrorMean_trimmed_m1 += zError_trimmed_m1;
      errorRMS_trimmed_m1 += error_trimmed_m1;
      goodPoints_trimmed_m1++;
    }
    if ( ! boost::math::isnan(error_trimmed_m2 ))
    {
      xErrorMean_trimmed_m2 += xError_trimmed_m2;
      yErrorMean_trimmed_m2 += yError_trimmed_m2;
      zErrorMean_trimmed_m2 += zError_trimmed_m2;
      errorRMS_trimmed_m2 += error_trimmed_m2;
      goodPoints_trimmed_m2++;
    }

  }
  MITK_INFO << "Method 1 Dumped " << numberOfPoints - goodPoints_m1 << " off screen points";
  MITK_INFO << "Method 2 Dumped " << numberOfPoints - goodPoints_m2 << " off screen points";
  xErrorMean_m1 /= goodPoints_m1;
  yErrorMean_m1 /= goodPoints_m1;
  zErrorMean_m1 /= goodPoints_m1;
  xErrorMean_m2 /= goodPoints_m2;
  yErrorMean_m2 /= goodPoints_m2;
  zErrorMean_m2 /= goodPoints_m2;
  errorRMS_m1 = sqrt(errorRMS_m1/goodPoints_m1);
  errorRMS_m2 = sqrt(errorRMS_m2/goodPoints_m2);
  
  MITK_INFO << "Mean x error c wrapper = " <<  xErrorMean_m1; 
  MITK_INFO << "Mean y error c wrapper = " <<  yErrorMean_m1; 
  MITK_INFO << "Mean z error c wrapper = " <<  zErrorMean_m1; 
  MITK_INFO << "RMS error c wrapper = " <<  errorRMS_m1; 
  MITK_INFO << "Mean x error c++ wrapper = " <<  xErrorMean_m2; 
  MITK_INFO << "Mean y error c++ wrapper = " <<  yErrorMean_m2; 
  MITK_INFO << "Mean z error c++ wrapper = " <<  zErrorMean_m2; 
  MITK_INFO << "RMS error c++ wrapper = " <<  errorRMS_m2; 
  MITK_TEST_CONDITION (fabs(xErrorMean_m1) < 0.5, "Testing x error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(yErrorMean_m1) < 0.5, "Testing y error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(zErrorMean_m1) < 0.5, "Testing z error mean value for c wrapper method");
  MITK_TEST_CONDITION (errorRMS_m1 < 2.0 , "Testing RMS error value for c method");
  MITK_TEST_CONDITION (fabs(xErrorMean_m2) < 1e-3, "Testing x error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(yErrorMean_m2) < 1e-3, "Testing y error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(zErrorMean_m2) < 1e-3, "Testing z error mean value for c++ method");
  MITK_TEST_CONDITION (errorRMS_m2 < 2e-3, "Testing RMS error value for c++ method");

  MITK_INFO << "Method 1 Trimmed Dumped " << numberOfPoints - goodPoints_trimmed_m1 << " off screen points";
  MITK_INFO << "Method 2 Trimmed Dumped " << numberOfPoints - goodPoints_trimmed_m2 << " off screen points";
  xErrorMean_trimmed_m1 /= goodPoints_trimmed_m1;
  yErrorMean_trimmed_m1 /= goodPoints_trimmed_m1;
  zErrorMean_trimmed_m1 /= goodPoints_trimmed_m1;
  xErrorMean_trimmed_m2 /= goodPoints_trimmed_m2;
  yErrorMean_trimmed_m2 /= goodPoints_trimmed_m2;
  zErrorMean_trimmed_m2 /= goodPoints_trimmed_m2;
  errorRMS_trimmed_m1 = sqrt(errorRMS_trimmed_m1/goodPoints_trimmed_m1);
  errorRMS_trimmed_m2 = sqrt(errorRMS_trimmed_m2/goodPoints_trimmed_m2);
  
  MITK_INFO << "Mean x error trimmed c wrapper = " <<  xErrorMean_trimmed_m1; 
  MITK_INFO << "Mean y error trimmed c wrapper = " <<  yErrorMean_trimmed_m1; 
  MITK_INFO << "Mean z error trimmed c wrapper = " <<  zErrorMean_trimmed_m1; 
  MITK_INFO << "RMS error trimmed c wrapper = " <<  errorRMS_trimmed_m1; 
  MITK_INFO << "Mean x error trimmed c++ wrapper = " <<  xErrorMean_trimmed_m2; 
  MITK_INFO << "Mean y error trimmed c++ wrapper = " <<  yErrorMean_trimmed_m2; 
  MITK_INFO << "Mean z error trimmed c++ wrapper = " <<  zErrorMean_trimmed_m2; 
  MITK_INFO << "RMS error trimmed c++ wrapper = " <<  errorRMS_trimmed_m2; 
  MITK_TEST_CONDITION (fabs(xErrorMean_trimmed_m1) < 0.5, "Testing x error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(yErrorMean_trimmed_m1) < 0.5, "Testing y error mean value for c wrapper method");
  MITK_TEST_CONDITION (fabs(zErrorMean_trimmed_m1) < 0.5, "Testing z error mean value for c wrapper method");
  MITK_TEST_CONDITION (errorRMS_trimmed_m1 < 2.0, "Testing RMS error value for c method");
  MITK_TEST_CONDITION (fabs(xErrorMean_trimmed_m2) < 1e-3, "Testing x error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(yErrorMean_trimmed_m2) < 1e-3, "Testing y error mean value for c++ method");
  MITK_TEST_CONDITION (fabs(zErrorMean_trimmed_m2) < 1e-3, "Testing z error mean value for c++ method");
  MITK_TEST_CONDITION (errorRMS_trimmed_m2 < 2e-3, "Testing RMS error value for c++ method");

  MITK_TEST_END();
}
