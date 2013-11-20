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
#include <mitkHandeyeCalibrate.h>
#include <mitkOpenCVMaths.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <niftkFileHelper.h>
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
bool CompareOpenCVMatrices2(cv::Mat mat1, cv::Mat mat2 , double tolerance) 
{
  assert ( mat1.size() == mat2.size() );
  assert ( mat1.type() == mat2.type() );
  cv::Mat absDiff = cv::Mat(mat1.size(),mat1.type());
  cv::Mat abs = cv::Mat(mat1.size(),mat1.type());
  cv::absdiff(mat1,mat2,absDiff);
  abs = cv::abs(mat1);
  cv::Scalar Sum1 = cv::sum (abs);
  cv::Scalar Sum = cv::sum (absDiff);
  MITK_INFO << std::endl << "Absolute difference Sum = " << Sum[0] << " : Normalised sum = " << Sum[0]/Sum1[0];
  if ( Sum[0]/Sum1[0] < tolerance ) 
  {
    return true;
  }
  else
  {
    return false;
  }
}
bool CompareOpenCVMatrices(CvMat* mat1, cv::Mat mat2, double tolerance)
{ 
  return CompareOpenCVMatrices2(cv::Mat(mat1), mat2,tolerance);
}
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
  double tolerance = 0.05;

  cv::Size imageSize;
  imageSize.height = screenHeight;
  imageSize.width = screenWidth;
  //the calibration grid pattern;
  int xcorners = 14;
  int ycorners = 10;
  double squareSize = 3; // in mm
  int maxTrackingMatrices = -1;

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
    if (( ok == false ) && (strcmp(argv[1],"--tolerance") == 0 ))
    {
      argc--;
      argv++;
      tolerance = atof(argv[1]);
      MITK_INFO << "Setting tolerance to  " << tolerance;
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
    
    if (( ok == false ) && (strcmp(argv[1],"--maxTrackingMatricesToUse") == 0 )) 
    {
      argc--;
      argv++;
      maxTrackingMatrices = atof(argv[1]);
      argc--;
      argv++;
      ok =true;
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
  std::string trackerDirectory = "testTrackerMatrices";
  niftk::CreateDirectoryAndParents(trackerDirectory);

  int views = MarkerToWorld.size(); 
  int stepSize = 1;

  if ( maxTrackingMatrices != -1 ) 
  {
    if ( maxTrackingMatrices < MarkerToWorld.size() )
    {
      stepSize = MarkerToWorld.size() / maxTrackingMatrices;
    }
    else
    {
      stepSize = 1;
    }
  }


  for ( unsigned int frame = 0 ; frame < views ; frame += stepSize ) 
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

      //and copy the matrix, (with added tracking noise?) to a subdirectory
      std::string trackerFilename = trackerDirectory + "/" + boost::lexical_cast<std::string>(allLeftImagePoints.size()) + ".txt";
      std::ofstream fs_tracker;
      fs_tracker.open(trackerFilename.c_str(), std::ios::out);
      for ( int row = 0 ; row < 4 ; row ++ )
      {
        for ( int col = 0 ; col < 4 ; col ++ )
        {
          fs_tracker << MarkerToWorld[frame].at<double>(row,col) << " " ;
        }
        fs_tracker << std::endl;
      }
      fs_tracker.close();


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
  CvMat* outputDistortionCoefficientsLeft = cvCreateMat(1,4,CV_64FC1);
  CvMat* outputRotationVectorsLeft = cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputTranslationVectorsLeft= cvCreateMat(allLeftImagePoints.size(),3,CV_64FC1);
  CvMat* outputIntrinsicMatrixRight= cvCreateMat(3,3,CV_64FC1);
  CvMat* outputDistortionCoefficientsRight= cvCreateMat(1,4,CV_64FC1);
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
    fs_leftIntrinsic << CV_MAT_ELEM (*outputDistortionCoefficientsLeft, double , 0, i ) << " ";
    fs_rightIntrinsic << CV_MAT_ELEM (*outputDistortionCoefficientsRight, double , 0, i ) << " ";
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


  //now do the handeye
  mitk::HandeyeCalibrate::Pointer handeyecalibrator = mitk::HandeyeCalibrate::New();
  handeyecalibrator->Calibrate(trackerDirectory,extrinsic);

  //Next need to do a bunch of tests, could start with just checking that the 
  //inputs approximately equal the outputs, but then could also do some proper 
  //reprojection tests.
  
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputIntrinsicMatrixLeft, leftCameraIntrinsic, tolerance), "Testing left intrinsic Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputDistortionCoefficientsLeft, leftCameraDistortion, tolerance), "Testing left distortion Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputIntrinsicMatrixRight, rightCameraIntrinsic, tolerance), "Testing right intrinsic Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputDistortionCoefficientsRight, rightCameraDistortion, tolerance), "Testing right distortion Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputDistortionCoefficientsRight, rightCameraDistortion, tolerance), "Testing right distortion Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputRightToLeftRotation, rightToLeftRotationMatrix, tolerance), "Testing right to left rotation Matrix"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices(outputRightToLeftTranslation, rightToLeftTranslationVector, tolerance), "Testing right to left translation vector"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices2(handeyecalibrator->GetCameraToMarker(), leftCameraToTracker, tolerance), "Testing handeye"); 
  MITK_TEST_CONDITION ( CompareOpenCVMatrices2(handeyecalibrator->GetGridToWorld(), gridToWorld, tolerance), "Testing grid to world"); 

  MITK_TEST_END();
}
