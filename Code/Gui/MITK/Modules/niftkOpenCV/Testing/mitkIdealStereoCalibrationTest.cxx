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
#include <mitkOpenCVMaths.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <fstream>

/**
 * Test for stereo calibration and handeye.
 * Start with the calibration grid, located somewhere in 3D space
 * Load in some camera calibration parameters, and some tracking matrices,
 * Project the grid points onto the camera, then use 
 * mitk::CalibrateStereoCameraParameters to reconstruct the calibration. 
 * Optionally then perform and handeye calibration.
 * Optionally add some tracking and or projection errors
 */


int mitkIdealStereoCalibrationTest ( int argc, char * argv[] )
{

  std::string calibrationDirectory = "";
  std::string gridToWorldTransform = "";
  std::string trackingMatrixDirectory = "";
  double pixelNoise = 0.0;
  double trackingNoise = 0.0;
  bool quantize = false;
  bool discardFramesWithNonVisiblePoints = true; 
  double screenWidth = 1980;
  double screenHeight = 540;

  cv::Size imageSize;
  imageSize.height = screenHeight;
  imageSize.width = screenWidth;
  //the calibration grid pattern;
  int xcorners = 14;
  int ycorners = 10;
  double squareSize = 3; // in mm

  cv::Mat gridToWorld = cvCreateMat (4,4,CV_64FC1);

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
    if (( ok == false ) && (strcmp(argv[1],"--tracking") == 0 ))
    {
      argc--;
      argv++;
      trackingMatrixDirectory = argv[1];
      MITK_INFO << "Loading tracking from " << trackingMatrixDirectory;
      argc--;
      argv++;
      ok=true;
    }
    if (( ok == false ) && (strcmp(argv[1],"--gridToWorldTransform")==0))
    {
      argc--;
      argv++; 
      std::ifstream fin(argv[1]);
      for ( int row = 0; row < 4; row ++ )
      {
        for ( int col = 0; col < 4; col ++ )
        {
          fin >> gridToWorld.at<double>(row,col);
        }
      }
      fin.close();
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
    if ( ok == false )
    {
      MITK_ERROR << "Failed to parse arguments";
      return EXIT_FAILURE;
    }
  }
      


  MITK_TEST_BEGIN("mitkIdealStereoCalibrationTest");

  //get the tracking matrices
  std::vector<cv::Mat> MarkerToWorld = mitk::LoadMatricesFromDirectory(trackingMatrixDirectory);
 
  //get the calibration data
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

  leftCameraPositionToFocalPointUnitVector.at<double>(0,0) = 0.0;
  leftCameraPositionToFocalPointUnitVector.at<double>(0,1) = 0.0;
  leftCameraPositionToFocalPointUnitVector.at<double>(0,2) = 1.0;


  std::vector<cv::Mat> allLeftImagePoints;
  std::vector<cv::Mat> allLeftObjectPoints;
  std::vector<cv::Mat> allRightImagePoints;
  std::vector<cv::Mat> allRightObjectPoints;

  cv::Mat worldPoints = cv::Mat (xcorners*ycorners,3,CV_64FC1);
  cv::Mat gridPoints = cv::Mat (xcorners*ycorners,3,CV_64FC1);
  cv::Mat leftCameraNormals = cv::Mat (xcorners*ycorners,3,CV_64FC1);

  for ( int row = 0 ; row < ycorners ; row ++ ) 
  {
    for ( int col = 0 ; col < xcorners ; col ++ )
    {
      cv::Point3d gridCorner = cv::Point3d ( row * squareSize, col * squareSize, 0 );
      cv::Point3d worldCorner;
      {
        using namespace mitk;
        worldCorner = gridToWorld * gridCorner;
      }
      worldPoints.at<double>(row * xcorners + col,0) = worldCorner.x;
      worldPoints.at<double>(row * xcorners + col,1) = worldCorner.y;
      worldPoints.at<double>(row * xcorners + col,2) = worldCorner.z;
      gridPoints.at<double>(row * xcorners + col,0) = gridCorner.x;
      gridPoints.at<double>(row * xcorners + col,1) = gridCorner.y;
      gridPoints.at<double>(row * xcorners + col,2) = gridCorner.z;
      leftCameraNormals.at<double>(row*xcorners + col,0) = 0.0;
      leftCameraNormals.at<double>(row*xcorners + col,1) = 0.0;
      leftCameraNormals.at<double>(row*xcorners + col,2) = -1.0;
    }
  }
  const int corner00 = 0;
  const int corner01 = 0 * xcorners + ycorners-1;
  const int corner10 = (ycorners-1)*xcorners;
  const int corner11 = (ycorners-1)*xcorners + (xcorners-1);

  MITK_INFO << "World Corners 1: (" << worldPoints.at<double>(corner00,0) << "," <<
    worldPoints.at<double>(corner00,1) << "," <<  worldPoints.at<double>(corner00,2) << ")(" <<
    worldPoints.at<double>(corner01,0) << "," << 
    worldPoints.at<double>(corner01,1) << "," <<
    worldPoints.at<double>(corner01,2) << ")";
  MITK_INFO << "World Corners 2: (" << 
    worldPoints.at<double>(corner10,0) << "," << 
    worldPoints.at<double>(corner10,1) << "," <<
    worldPoints.at<double>(corner10,2) << ")(" <<
    worldPoints.at<double>(corner11,0) << "," << 
    worldPoints.at<double>(corner11,1) << "," << 
    worldPoints.at<double>(corner11,2) << ")";
  for ( unsigned int frame = 0 ; frame < MarkerToWorld.size() ; frame ++ ) 
  {

    //get world points into camera lens coordinates
    cv::Mat leftCameraPoints = cv::Mat (xcorners*ycorners,3,CV_64FC1);
    cv::Mat worldToCamera = (MarkerToWorld[frame] * leftCameraToTracker).inv();

    for ( int j = 0 ; j < xcorners*ycorners ; j ++ ) 
    {
      cv::Point3d worldCorner ; 
      cv::Point3d leftCameraCorner;
      worldCorner.x = worldPoints.at<double>(j,0);
      worldCorner.y = worldPoints.at<double>(j,1);
      worldCorner.z = worldPoints.at<double>(j,2);

      {
        using namespace mitk;
        leftCameraCorner = worldToCamera * worldCorner;
      }
      leftCameraPoints.at<double>(j,0) = leftCameraCorner.x;
      leftCameraPoints.at<double>(j,1) = leftCameraCorner.y;
      leftCameraPoints.at<double>(j,2) = leftCameraCorner.z;
    }
    MITK_INFO << frame << " Left Cam Corners 1: (" << leftCameraPoints.at<double>(corner00,0) << "," <<
      leftCameraPoints.at<double>(corner00,1) << "," <<  leftCameraPoints.at<double>(corner00,2) << ")(" <<
      leftCameraPoints.at<double>(corner01,0) << "," << 
      leftCameraPoints.at<double>(corner01,1) << "," <<
      leftCameraPoints.at<double>(corner01,2) << ")";
    MITK_INFO << frame << " Left Cam Corners 2: (" << 
      leftCameraPoints.at<double>(corner10,0) << "," << 
      leftCameraPoints.at<double>(corner10,1) << "," <<
      leftCameraPoints.at<double>(corner10,2) << ")(" <<
      leftCameraPoints.at<double>(corner11,0) << "," << 
      leftCameraPoints.at<double>(corner11,1) << "," << 
      leftCameraPoints.at<double>(corner11,2) << ")";
 
    //project onto screen
    CvMat* outputLeftCameraWorldPointsIn3D = NULL;
    CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
    CvMat* output2DPointsLeft = NULL ;
    CvMat* output2DPointsRight = NULL;
//

    //project points
    std::vector<int> Points = mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraPoints,leftCameraNormals,
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
    for ( int j = 0 ; j < xcorners*ycorners ; j ++ ) 
    {
      CV_MAT_ELEM (*output2DPointsLeft ,double,j,0) += var_nor(); 
      CV_MAT_ELEM (*output2DPointsLeft ,double,j,1) += var_nor(); 
      CV_MAT_ELEM (*output2DPointsRight ,double,j,0) += var_nor(); 
      CV_MAT_ELEM (*output2DPointsRight ,double,j,1) += var_nor(); 
    }

    if ( quantize ) 
    {
      for ( int j = 0 ; j < xcorners*ycorners ; j ++ ) 
      {
        CV_MAT_ELEM (*output2DPointsLeft ,double,j,0) =
            floor ( CV_MAT_ELEM (*output2DPointsLeft ,double,j,0) + 0.5 );
        CV_MAT_ELEM (*output2DPointsLeft ,double,j,1) =
            floor ( CV_MAT_ELEM (*output2DPointsLeft ,double,j,1) + 0.5 );
        CV_MAT_ELEM (*output2DPointsRight ,double,j,0) =
            floor ( CV_MAT_ELEM (*output2DPointsRight ,double,j,0) + 0.5 );
        CV_MAT_ELEM (*output2DPointsRight ,double,j,1) =
            floor ( CV_MAT_ELEM (*output2DPointsRight ,double,j,1) + 0.5 );
      }
    }

    //check if all points are on screen
    bool allOnScreen = true;
    for ( int j = 0 ; j < xcorners*ycorners ; j++ )
    {
      if (
          CV_MAT_ELEM (*output2DPointsLeft ,double,j,0) < 0 ||
          CV_MAT_ELEM (*output2DPointsLeft ,double,j,0) > screenWidth 
          || CV_MAT_ELEM (*output2DPointsLeft ,double,j,1) < 0 || 
          CV_MAT_ELEM (*output2DPointsLeft ,double,j,1) > screenHeight ||
          CV_MAT_ELEM (*output2DPointsRight ,double,j,0) < 0 ||
          CV_MAT_ELEM (*output2DPointsRight ,double,j,0) > screenWidth 
          || CV_MAT_ELEM (*output2DPointsRight ,double,j,1) < 0 || 
          CV_MAT_ELEM (*output2DPointsRight ,double,j,1) > screenHeight
          )
      {
        allOnScreen = false;
        j=xcorners*ycorners;
      }
    }
    
    if ( allOnScreen )
    {
      //add to our point sets
      allLeftImagePoints.push_back(output2DPointsLeft);
      allLeftObjectPoints.push_back(gridPoints);
      allRightImagePoints.push_back(output2DPointsRight);
      allRightObjectPoints.push_back(gridPoints);
    }
    
  }

  MITK_INFO << "There are " << allLeftImagePoints.size() << " good frames";
  cv::Mat leftImagePoints (xcorners * ycorners * allLeftImagePoints.size(),2,CV_64FC1);
  cv::Mat leftObjectPoints (xcorners * ycorners * allLeftImagePoints.size(),3,CV_64FC1);
  cv::Mat rightImagePoints (xcorners * ycorners * allLeftImagePoints.size(),2,CV_64FC1);
  cv::Mat rightObjectPoints (xcorners * ycorners * allLeftImagePoints.size(),3,CV_64FC1);

  cv::Mat leftPointCounts (allLeftImagePoints.size(),1,CV_32SC1);
  cv::Mat rightPointCounts (allLeftImagePoints.size(),1,CV_32SC1);



  for ( unsigned int i = 0 ; i < allLeftImagePoints.size() ; i++ )
  {
    MITK_INFO << "Filling "  << i;
    for ( unsigned int j = 0 ; j < xcorners*ycorners ; j ++ )
    {
      leftImagePoints.at<double>(i* xcorners * ycorners + j,0) =
        allLeftImagePoints[i].at<double>(j,0);
      leftImagePoints.at<double>(i* xcorners * ycorners + j,1) =
        allLeftImagePoints[i].at<double>(j,1);
      leftObjectPoints.at<double>(i* xcorners * ycorners + j,0) =
        allLeftObjectPoints[i].at<double>(j,0);
      leftObjectPoints.at<double>(i* xcorners * ycorners + j,1) =
        allLeftObjectPoints[i].at<double>(j,1);
      leftObjectPoints.at<double>(i* xcorners * ycorners + j,2) =
        allLeftObjectPoints[i].at<double>(j,2);
      rightImagePoints.at<double>(i* xcorners * ycorners + j,0) =
        allRightImagePoints[i].at<double>(j,0);
      rightImagePoints.at<double>(i* xcorners * ycorners + j,1) =
        allRightImagePoints[i].at<double>(j,1);
      rightObjectPoints.at<double>(i* xcorners * ycorners + j,0) =
        allRightObjectPoints[i].at<double>(j,0);
      rightObjectPoints.at<double>(i* xcorners * ycorners + j,1) =
        allRightObjectPoints[i].at<double>(j,1);
      rightObjectPoints.at<double>(i* xcorners * ycorners + j,2) =
        allRightObjectPoints[i].at<double>(j,2);
    }
    leftPointCounts.at<int>(i,0) = xcorners * ycorners;
    rightPointCounts.at<int>(i,0) = xcorners * ycorners;
  }

  MITK_INFO << "Starting intrinisic calibration";
  CvMat* outputIntrinsicMatrixLeft = cvCreateMat(3,3,CV_64FC1);
  CvMat* outputDistortionCoefficientsLeft = cvCreateMat(4,1,CV_64FC1);
  CvMat* outputRotationVectorsLeft = cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputTranslationVectorsLeft= cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputIntrinsicMatrixRight= cvCreateMat(3,3,CV_64FC1);
  CvMat* outputDistortionCoefficientsRight= cvCreateMat(4,1,CV_64FC1);
  CvMat* outputRotationVectorsRight= cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputTranslationVectorsRight= cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputRightToLeftRotation = cvCreateMat(3,3,CV_64FC1);
  CvMat* outputRightToLeftTranslation = cvCreateMat(3,1,CV_64FC1);
  CvMat* outputEssentialMatrix = cvCreateMat(3,3,CV_64FC1);
  CvMat* outputFundamentalMatrix= cvCreateMat(3,3,CV_64FC1);

  mitk::CalibrateStereoCameraParameters(
    leftObjectPoints,
    leftImagePoints,
    leftPointCounts,
    imageSize,
    rightObjectPoints,
    rightImagePoints,
    rightPointCounts,
    *outputIntrinsicMatrixLeft,
    *outputDistortionCoefficientsLeft,
    *outputRotationVectorsLeft,
    *outputTranslationVectorsLeft,
    *outputIntrinsicMatrixRight,
    *outputDistortionCoefficientsRight,
    *outputRotationVectorsRight,
    *outputTranslationVectorsRight,
    *outputRightToLeftRotation,
    *outputRightToLeftTranslation,
    *outputEssentialMatrix,
    *outputFundamentalMatrix);

  //write it out
  std::string leftIntrinsic = "test.calib.left.intrinsic.txt";
  std::string rightIntrinsic = "test.calib.right.intrinsic.txt";
  std::string rightToLeft = "test.calib.r2l.txt";
  std::string extrinsic = "test.leftextrinsics.txt";

  std::ofstream fs_leftIntrinsic;
  std::ofstream fs_rightIntrinsic;
  std::ofstream fs_r2l;
  std::ofstream fs_ext;

  fs_leftIntrinsic.open(leftIntrinsic.c_str(), std::ios::out);
  fs_rightIntrinsic.open(rightIntrinsic.c_str(), std::ios::out);
  fs_r2l.open(rightToLeft.c_str(), std::ios::out);
  fs_ext.open(extrinsic.c_str(), std::ios::out);

  for ( int row = 0 ; row < 3 ; row ++ ) 
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      fs_leftIntrinsic << CV_MAT_ELEM (*outputIntrinsicMatrixLeft, double, row,col) << " ";
      fs_rightIntrinsic << CV_MAT_ELEM (*outputIntrinsicMatrixRight, double, row,col) << " ";
      fs_r2l << CV_MAT_ELEM (*outputRightToLeftRotation, double , row,col) << " ";
    }
    fs_leftIntrinsic << std::endl;
    fs_rightIntrinsic << std::endl;
    fs_r2l << std::endl;
  }
  for ( int i = 0 ; i < 4 ; i ++ )  
  {
    fs_leftIntrinsic << CV_MAT_ELEM (*outputDistortionCoefficientsLeft, double , i, 0 ) << " ";
    fs_rightIntrinsic << CV_MAT_ELEM (*outputDistortionCoefficientsRight, double , i, 0 ) << " ";
  }
  for ( int i = 0 ; i < 3 ; i ++ )  
  {
    fs_r2l << CV_MAT_ELEM (*outputRightToLeftTranslation, double , i, 0 ) << " ";
  }

  fs_leftIntrinsic.close();
  fs_rightIntrinsic.close();
  fs_r2l.close();
  for ( unsigned int view = 0 ; view < allLeftImagePoints.size() ; view ++ )
  {
    for ( int i = 0 ; i < 3 ; i ++ )
    {
      fs_ext << CV_MAT_ELEM ( *outputRotationVectorsLeft , double  , view, i) << " ";
    }
    for ( int i = 0 ; i < 3 ; i ++ )
    {
      fs_ext << CV_MAT_ELEM ( *outputTranslationVectorsLeft , double  , view, i) << " ";
    }
    fs_ext << std::endl;
  }
  fs_ext.close();
 /*   mitk::UndistortPoints(output2DPointsLeft, 
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
    mitk::TriangulatePointPairs(
        inputUndistortedPoints, 
        leftCameraIntrinsic,
        rightCameraIntrinsic,
        rightToLeftRotationVector,
        rightToLeftTranslationVector);

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
  MITK_TEST_CONDITION (errorRMS_m2 < 2.0 , "Testing RMS error value for c++ method");*/

  MITK_TEST_END();
}
