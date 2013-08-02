/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkHandeyeCalibrateFromDirectory.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <FileHelper.h>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrateFromDirectory::HandeyeCalibrateFromDirectory()
: m_FlipTracking(true)
, m_FlipExtrinsic(false)
, m_SortByDistance(false)
, m_SortByAngle(false)
, m_FramesToUse(40)
, m_BadFrameFactor(2.0)
, m_SaveProcessedVideoData(true)
, m_VideoInitialised(false)
, m_TrackingDataInitialised(false)
, m_TrackerIndex(0)
, m_AbsTrackerTimingError(20e6) // 20 milliseconds
, m_NumberCornersWidth(14)
, m_NumberCornersHeight(10)
, m_SquareSizeInMillimetres(3.0)
, m_WriteOutChessboards(false)
{
}


//-----------------------------------------------------------------------------
HandeyeCalibrateFromDirectory::~HandeyeCalibrateFromDirectory()
{

}

//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::InitialiseVideo()
{
  std::vector<std::string> filenames = FindVideoData();
  if ( filenames.size() == 0 ) 
  {
    MITK_ERROR << "Failed to find any video files";
    m_VideoInitialised = false;
    return;
  }
  if ( filenames.size() > 1 ) 
  {
    MITK_ERROR << "Found too many video files. ";
    for ( unsigned int  i = 0 ; i < filenames.size() ; i++ )
    {
      MITK_ERROR << filenames[i];
    }
    m_VideoInitialised = false;
    return;
  }

  MITK_INFO << "Loading video frames from " << filenames[0];
  LoadVideoData (filenames[0]);
  return;
    
}

//-----------------------------------------------------------------------------
std::vector<std::string> HandeyeCalibrateFromDirectory::FindVideoData()
{
  boost::filesystem::recursive_directory_iterator end_itr;
  std::vector<std::string> returnStrings;

  for ( boost::filesystem::recursive_directory_iterator it(m_Directory);
            it != end_itr ; ++it)
  {
    if (  it->path().extension() == ".264" )
    {
      returnStrings.push_back(it->path().string());
    }
  }
  return returnStrings;
}

//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::LoadVideoData(std::string filename)
{
  cv::VideoCapture capture = cv::VideoCapture(filename) ; 
  
  if ( ! capture.isOpened() ) 
  {
    MITK_ERROR << "Failed to open " << filename;
    return;
  }
  
  if ( m_Matcher.IsNull() )
  {
    MITK_INFO << "Initialising Video Tracker Matcher";
    m_Matcher = mitk::VideoTrackerMatching::New();
  }
  if ( ! m_Matcher->IsReady() )
  {
    m_Matcher->Initialise (m_Directory);
  }

  //get frame count doesn't work for 264 files, which are just 
  //raw data get the frame count from the framemap log
  int numberOfFrames = m_Matcher->GetNumberOfFrames();
  double framewidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  double frameheight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  
  double filesize = numberOfFrames * framewidth * frameheight * 4 / 1e9;
  MITK_INFO << numberOfFrames << "frames in video : " << framewidth << "x" << frameheight;;
  MITK_INFO << filesize << "gigabytes required to store";
  
  //go through data set, randomly selecting frames from a uniform distribution until 
  //sufficient frames are found.
  //First check tracking data is ok (based on timing error threshold)
  //Then try and extract corners in both left and right frames.
  //std::default_random_engine generator;
  //std::uniform_int_distribution<int> distribution (0, numberOfFrames/2);

  std::srand(0);
  std::vector <int> LeftFramesToUse;
  std::vector <int> RightFramesToUse;
  while ( LeftFramesToUse.size() < m_FramesToUse * m_BadFrameFactor )
  {
    int FrameToUse =  std::rand()%(numberOfFrames/2);
    //first check it's not already in array
    if ( (std::find(LeftFramesToUse.begin(), LeftFramesToUse.end(), FrameToUse * 2 ) == LeftFramesToUse.end()) ) 
    {
      MITK_INFO << "Trying frame pair " << FrameToUse * 2 << "," << FrameToUse*2 +1;
    
      long long int*  LeftTimingError = new long long;
      long long int *  RightTimingError = new long long;
      cv::Mat LeftTrackingMatrix = m_Matcher->GetTrackerMatrix(FrameToUse * 2 , 
        LeftTimingError, m_TrackerIndex );
      cv::Mat RightTrackingMatrix = m_Matcher->GetTrackerMatrix(FrameToUse * 2 + 1 , 
        RightTimingError, m_TrackerIndex );
      if ( std::abs(*LeftTimingError) > m_AbsTrackerTimingError ||
        std::abs(*RightTimingError) > m_AbsTrackerTimingError ) 
      {
        MITK_INFO << "Rejecting frame " << FrameToUse << "Due to high timing error: " <<
          std::abs(*LeftTimingError) << " > " <<  m_AbsTrackerTimingError;
      }
      else
      {
        //timing error OK, now check if we can extract corners
     
        LeftFramesToUse.push_back (FrameToUse *2);
        RightFramesToUse.push_back (FrameToUse * 2 + 1);
      }
    }
  }
  //now go through video and extract frames to use
  int FrameNumber = 0 ;

  std::vector<cv::Mat>  allLeftImagePoints;
  std::vector<cv::Mat>  allLeftObjectPoints;
  std::vector<cv::Mat>  allRightImagePoints;
  std::vector<cv::Mat>  allRightObjectPoints;

  cv::Size imageSize;

  while ( FrameNumber < numberOfFrames )
  {
    cv::Mat TempFrame;
    cv::Mat LeftFrame;
    cv::Mat RightFrame;
    capture >> TempFrame;
    LeftFrame = TempFrame.clone();
    capture >> TempFrame;
    RightFrame = TempFrame.clone();
    imageSize=RightFrame.size();
    if ( (std::find(LeftFramesToUse.begin(), LeftFramesToUse.end(), FrameNumber) != LeftFramesToUse.end()) ) 
    {
      if ((std::find(RightFramesToUse.begin(),RightFramesToUse.end(),FrameNumber + 1) != RightFramesToUse.end()) )
      { 

        MITK_INFO << "Using frame pair" << FrameNumber << "," <<FrameNumber+1;
        std::vector <cv::Point2f>* leftImageCorners = new std::vector<cv::Point2f>;
        std::vector <cv::Point3f>* leftObjectCorners = new std::vector<cv::Point3f>;
        mitk::ExtractChessBoardPoints (
          LeftFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres,
          leftImageCorners, leftObjectCorners);
        std::vector <cv::Point2f>* rightImageCorners = new std::vector<cv::Point2f>;
        std::vector <cv::Point3f>* rightObjectCorners = new std::vector<cv::Point3f>;
        mitk::ExtractChessBoardPoints (
          RightFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres,
          rightImageCorners, rightObjectCorners);

        if ( leftImageCorners->size() == m_NumberCornersWidth * m_NumberCornersHeight &&
            rightImageCorners->size() == m_NumberCornersWidth * m_NumberCornersHeight )
        {

          MITK_INFO << "Frame " << capture.get(CV_CAP_PROP_POS_FRAMES)-2 << " got " << leftImageCorners->size() << " corners for both images";
          allLeftImagePoints.push_back(cv::Mat(*leftImageCorners,true));
          allLeftObjectPoints.push_back(cv::Mat(*leftObjectCorners,true));
          allRightImagePoints.push_back(cv::Mat(*rightImageCorners,true));
          allRightObjectPoints.push_back(cv::Mat(*rightObjectCorners,true));
        }
        else
        {
          MITK_INFO << "Frame " <<  capture.get(CV_CAP_PROP_POS_FRAMES)-2 << " failed corner extraction. Removing from good frame buffer [" << leftImageCorners->size() << "," << rightImageCorners->size() << "].";
          std::vector<int>::iterator newEnd = std::remove(LeftFramesToUse.begin(), LeftFramesToUse.end(), FrameNumber);
          LeftFramesToUse.erase(newEnd, LeftFramesToUse.end());
          newEnd = std::remove(RightFramesToUse.begin(), RightFramesToUse.end(), FrameNumber+1);
          RightFramesToUse.erase(newEnd, RightFramesToUse.end());
        }
        
        if ( m_WriteOutChessboards )
        {
          std::string leftfilename = m_Directory + "/LeftFrame" + boost::lexical_cast<std::string>(FrameNumber) + ".jpg";
          std::string rightfilename = m_Directory + "/RightFrame" + boost::lexical_cast<std::string>(FrameNumber + 1) + ".jpg";
          MITK_INFO << "Writing image to " << leftfilename;
          cv::imwrite( leftfilename, LeftFrame );
          cv::imwrite( rightfilename, RightFrame );
        }
      }
      else
      {
        MITK_ERROR << "Left right frame mismatch" ;
        return;
      }
    }
    
    FrameNumber++;
    FrameNumber++;
  }
  MITK_INFO << "There are " << LeftFramesToUse.size() << " good frames";

  cv::Mat leftImagePoints (m_NumberCornersWidth * m_NumberCornersHeight * LeftFramesToUse.size(),2,CV_32FC1);
  cv::Mat leftObjectPoints (m_NumberCornersWidth * m_NumberCornersHeight * LeftFramesToUse.size(),3,CV_32FC1);
  cv::Mat rightImagePoints (m_NumberCornersWidth * m_NumberCornersHeight * LeftFramesToUse.size(),2,CV_32FC1);
  cv::Mat rightObjectPoints (m_NumberCornersWidth * m_NumberCornersHeight * LeftFramesToUse.size(),3,CV_32FC1);
  
  cv::Mat leftPointCounts (LeftFramesToUse.size(),1,CV_32SC1);
  cv::Mat rightPointCounts (LeftFramesToUse.size(),1,CV_32SC1);

  if  (  allLeftImagePoints.size() !=  LeftFramesToUse.size() || 
           allLeftObjectPoints.size() !=  LeftFramesToUse.size() || 
           allRightImagePoints.size() !=  LeftFramesToUse.size() || 
           allRightObjectPoints.size() !=  LeftFramesToUse.size() )
  {
    MITK_ERROR << "Detected unequal matrix sizes";
    return;
  }
  for ( unsigned int i = 0 ; i < LeftFramesToUse.size() ; i ++  ) 
  {
    unsigned int size1 = allLeftImagePoints[i].size().height;
    //FIX ME
    unsigned int size2 = allLeftObjectPoints[0].size().height;
    unsigned int size3 = allRightImagePoints[i].size().height;
    unsigned int size4 = allRightObjectPoints[i].size().height;
    MITK_INFO << i << " " << size1 << ", " << size2 << ", " << size3 << ", " << size4;
  
    if ( size1 != m_NumberCornersWidth * m_NumberCornersHeight ||
          size2 != m_NumberCornersWidth * m_NumberCornersHeight ||
          size3 != m_NumberCornersWidth * m_NumberCornersHeight ||
          size4 != m_NumberCornersWidth * m_NumberCornersHeight)
    {
      MITK_ERROR << "Detected unequal matrix sizes";
      return;
    }
  }


  for ( unsigned int i = 0 ; i < LeftFramesToUse.size() ; i++ )
  {
    MITK_INFO << "Filling "  << i;
    for ( unsigned int j = 0 ; j < m_NumberCornersWidth * m_NumberCornersHeight ; j ++ ) 
    {
      leftImagePoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        allLeftImagePoints[i].at<float>(j,0);
      leftImagePoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        allLeftImagePoints[i].at<float>(j,1);
     
      leftObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        allLeftObjectPoints[0].at<float>(j,0);
      leftObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        allLeftObjectPoints[0].at<float>(j,1);
      leftObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,2) =
        allLeftObjectPoints[0].at<float>(j,2);

      rightImagePoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        allRightImagePoints[i].at<float>(j,0);
      rightImagePoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        allRightImagePoints[i].at<float>(j,1);
      
      rightObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        allRightObjectPoints[i].at<float>(j,0);
      rightObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        allRightObjectPoints[i].at<float>(j,1);
      rightObjectPoints.at<float>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,2) =
        allRightObjectPoints[i].at<float>(j,2);

    }
    leftPointCounts.at<int>(i,0) = m_NumberCornersWidth * m_NumberCornersHeight;
    rightPointCounts.at<int>(i,0) = m_NumberCornersWidth * m_NumberCornersHeight;
  }
  if ( true )
  {
    std::string leftimagePointsfilename = m_Directory + "/LeftImagePoints.xml";
    std::string leftObjectPointsfilename = m_Directory + "/LeftObjectPoints.xml";
    std::string rightimagePointsfilename = m_Directory + "/RightImagePoints.xml";
    std::string rightObjectPointsfilename = m_Directory + "/RightObjectPoints.xml";
    std::string leftPointCountfilename = m_Directory + "/LeftPointCount.xml";
    std::string rightPointCountfilename = m_Directory + "/RightPointCount.xml";
    cv::FileStorage fs1;
    fs1.open(leftimagePointsfilename, cv::FileStorage::WRITE);
    cv::FileStorage fs2(leftObjectPointsfilename, cv::FileStorage::WRITE);
    cv::FileStorage fs3(rightimagePointsfilename, cv::FileStorage::WRITE);
    cv::FileStorage fs4(rightObjectPointsfilename, cv::FileStorage::WRITE);
    cv::FileStorage fs5(leftPointCountfilename, cv::FileStorage::WRITE);
    cv::FileStorage fs6(rightPointCountfilename, cv::FileStorage::WRITE);
    fs1 <<  "leftimagepoints" << leftImagePoints;
    fs2 <<  "leftobjectpoints" << leftObjectPoints;
    fs3 <<  "rightimagepoints" << rightImagePoints;
    fs4 <<  "rightobjectpoints" << rightObjectPoints;
    fs5 <<  "leftpointcounts" << leftPointCounts;
    fs6 <<  "rightpointcounts" << rightPointCounts;
    fs1.release();
    fs2.release();
    fs3.release();
    fs4.release();
    fs5.release();
    fs6.release();
  }

  MITK_INFO << "Starting intrinisic calibration";
  CvMat* outputIntrinsicMatrixLeft = cvCreateMat(3,3,CV_32FC1);
  CvMat* outputDistortionCoefficientsLeft = cvCreateMat(5,1,CV_32FC1);
  CvMat* outputRotationVectorsLeft = cvCreateMat(LeftFramesToUse.size(),3,CV_32FC1);
  CvMat* outputTranslationVectorsLeft= cvCreateMat(LeftFramesToUse.size(),3,CV_32FC1);
  CvMat* outputIntrinsicMatrixRight= cvCreateMat(3,3,CV_32FC1);
  CvMat* outputDistortionCoefficientsRight= cvCreateMat(5,1,CV_32FC1);
  CvMat* outputRotationVectorsRight= cvCreateMat(LeftFramesToUse.size(),3,CV_32FC1);
  CvMat* outputTranslationVectorsRight= cvCreateMat(LeftFramesToUse.size(),3,CV_32FC1);
  CvMat* outputRightToLeftRotation = cvCreateMat(3,3,CV_32FC1);
  CvMat* outputRightToLeftTranslation = cvCreateMat(3,1,CV_32FC1);
  CvMat* outputEssentialMatrix = cvCreateMat(3,3,CV_32FC1);
  CvMat* outputFundamentalMatrix= cvCreateMat(3,3,CV_32FC1);


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
  std::string leftIntrinsic = m_Directory + "/calib.left.intrinsic";
  std::string rightIntrinsic = m_Directory + "/calib.right.intrinsic";
  std::string rightToLeft = m_Directory + "/calib.r2l";
  std::string extrinsic = m_Directory + "/leftextrinsics.txt";

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
      fs_leftIntrinsic << CV_MAT_ELEM (*outputIntrinsicMatrixLeft, float, row,col) << " ";
      fs_rightIntrinsic << CV_MAT_ELEM (*outputIntrinsicMatrixRight, float, row,col) << " ";
      fs_r2l << CV_MAT_ELEM (*outputRightToLeftRotation, float , row,col);
    }
    fs_leftIntrinsic << std::endl;
    fs_rightIntrinsic << std::endl;
    fs_r2l << std::endl;
  }
  for ( int i = 0 ; i < 5 ; i ++ )  
  {
    fs_leftIntrinsic << CV_MAT_ELEM (*outputDistortionCoefficientsLeft, float , i, 0 ) << " ";
  }
  for ( int i = 0 ; i < 3 ; i ++ )  
  {
    fs_r2l << CV_MAT_ELEM (*outputRightToLeftTranslation, float , i, 0 ) << " ";
  }

  fs_leftIntrinsic.close();
  fs_rightIntrinsic.close();
  fs_r2l.close();
  for ( unsigned int view = 0 ; view < LeftFramesToUse.size() ; view ++ )
  {
    for ( int i = 0 ; i < 3 ; i ++ ) 
    {
      fs_ext << CV_MAT_ELEM ( *outputRotationVectorsLeft , float  , view, i);
    }
    for ( int i = 0 ; i < 3 ; i ++ ) 
    {
      fs_ext << CV_MAT_ELEM ( *outputTranslationVectorsLeft , float  , view, i);
    }
    
    cv::Mat LeftTrackingMatrix = m_Matcher->GetTrackerMatrix(LeftFramesToUse[view] , 
        NULL, m_TrackerIndex );

    std::string trackerFilename = m_Directory + "/TrackerMatrices" + boost::lexical_cast<std::string>(m_TrackerIndex) + "/" + boost::lexical_cast<std::string>(view) + ".txt";
    
    std::ofstream fs_tracker;
    fs_tracker.open(trackerFilename.c_str(), std::ios::out);
    for ( int row = 0 ; row < 4 ; row ++ ) 
    {
      for ( int col = 0 ; col < 4 ; col ++ ) 
      {
        fs_tracker << LeftTrackingMatrix.at<double>(row,col) << " " ;
      }
      fs_tracker << std::endl;
    }
    fs_tracker.close();
  }
  fs_ext.close();

  Calibrate ( m_Directory + "/TrackerMatrices" + boost::lexical_cast<std::string>(m_TrackerIndex) , extrinsic); 
}
//-----------------------------------------------------------------------------
std::vector<double> HandeyeCalibrateFromDirectory::Calibrate(const std::string& TrackingFileDirectory,
  const std::string& ExtrinsicFileDirectoryOrFile,
  const std::string GroundTruthSolution)
{

  std::vector<cv::Mat> MarkerToWorld = mitk::LoadMatricesFromDirectory(TrackingFileDirectory);
  std::vector<cv::Mat> GridToCamera;
  std::vector<double> residuals;
  //init residuals with negative number to stop unit test passing
  //if Load result and calibration both produce zero.
  residuals.push_back(-100.0);
  residuals.push_back(-100.0);

  if ( niftk::DirectoryExists ( ExtrinsicFileDirectoryOrFile ))
  {
    GridToCamera = mitk::LoadOpenCVMatricesFromDirectory(ExtrinsicFileDirectoryOrFile);
  }
  else
  {
    GridToCamera = mitk::LoadMatricesFromExtrinsicFile(ExtrinsicFileDirectoryOrFile);
  }

  if ( MarkerToWorld.size() != GridToCamera.size() )
  {
    std::cerr << "ERROR: Called HandeyeCalibrate with unequal number of views and tracking matrices" << std::endl;
    return residuals;
  }
  int NumberOfViews = MarkerToWorld.size();
 

  if ( m_FlipTracking )
  {
    MarkerToWorld = mitk::FlipMatrices(MarkerToWorld);
  }
  if ( m_FlipExtrinsic )
  {
    GridToCamera = mitk::FlipMatrices(GridToCamera);
  }

  std::vector<int> indexes;
  //if SortByDistance and SortByAngle are both true, we'll sort by distance only
  if ( m_SortByDistance )
  {
    indexes = mitk::SortMatricesByDistance(MarkerToWorld);
    std::cout << "Sorted by distances " << std::endl;
  }
  else
  {
    if ( m_SortByAngle )
    {
      indexes = mitk::SortMatricesByAngle(MarkerToWorld);
      std::cout << "Sorted by angles " << std::endl;
    }
    else
    {
      for ( unsigned int i = 0; i < MarkerToWorld.size(); i ++ )
      {
        indexes.push_back(i);
      }
      std::cout << "No Sorting" << std::endl;
    }
  }

  for ( unsigned int i = 0; i < indexes.size(); i++ )
  {
    std::cout << indexes[i] << " ";
  }
  std::cout << std::endl;

  std::vector<cv::Mat> SortedGridToCamera;
  std::vector<cv::Mat> SortedMarkerToWorld;

  for ( unsigned int i = 0; i < indexes.size(); i ++ )
  {
    SortedGridToCamera.push_back(GridToCamera[indexes[i]]);
    SortedMarkerToWorld.push_back(MarkerToWorld[indexes[i]]);
  }

  cv::Mat A = cvCreateMat ( 3 * (NumberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (NumberOfViews - 1), 1, CV_64FC1 );

  for ( int i = 0; i < NumberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = SortedMarkerToWorld[i+1].inv() * SortedMarkerToWorld[i];
    mat2 = SortedGridToCamera[i+1] * SortedGridToCamera[i].inv();

    cv::Mat rotationMat1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationMat2 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationVector1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat rotationVector2 = cvCreateMat(3,1,CV_64FC1);
    for ( int row = 0; row < 3; row ++ )
    {
      for ( int col = 0; col < 3; col ++ )
      {
        rotationMat1.at<double>(row,col) = mat1.at<double>(row,col);
        rotationMat2.at<double>(row,col) = mat2.at<double>(row,col);
      }
    }
    cv::Rodrigues (rotationMat1, rotationVector1 );
    cv::Rodrigues (rotationMat2, rotationVector2 );

    double norm1 = cv::norm(rotationVector1);
    double norm2 = cv::norm(rotationVector2);

    rotationVector1 *= 2*sin(norm1/2) / norm1;
    rotationVector2 *= 2*sin(norm2/2) / norm2;

    cv::Mat sum = rotationVector1 + rotationVector2;
    cv::Mat diff = rotationVector2 - rotationVector1;

    A.at<double>(i*3+0,0)=0.0;
    A.at<double>(i*3+0,1)=-(sum.at<double>(2,0));
    A.at<double>(i*3+0,2)=sum.at<double>(1,0);
    A.at<double>(i*3+1,0)=sum.at<double>(2,0);
    A.at<double>(i*3+1,1)=0.0;
    A.at<double>(i*3+1,2)=-(sum.at<double>(0,0));
    A.at<double>(i*3+2,0)=-(sum.at<double>(1,0));
    A.at<double>(i*3+2,1)=sum.at<double>(0,0);
    A.at<double>(i*3+2,2)=0.0;
 
    b.at<double>(i*3+0,0)=diff.at<double>(0,0);
    b.at<double>(i*3+1,0)=diff.at<double>(1,0);
    b.at<double>(i*3+2,0)=diff.at<double>(2,0);
  
  }
  
  cv::Mat PseudoInverse = cvCreateMat(3,3,CV_64FC1);
  cv::invert(A,PseudoInverse,CV_SVD);
 
  cv::Mat pcgPrime = PseudoInverse * b;

  cv::Mat Error = A * pcgPrime-b;
 
  cv::Mat ErrorTransMult = cvCreateMat(Error.cols, Error.cols, CV_64FC1);
 
  cv::mulTransposed (Error, ErrorTransMult, true);
     
  double RotationResidual = sqrt(ErrorTransMult.at<double>(0,0)/(NumberOfViews-1));
  residuals[0] = RotationResidual;
 
  cv::Mat pcg = 2 * pcgPrime / ( sqrt(1 + cv::norm(pcgPrime) * cv::norm(pcgPrime)) );
  cv::Mat id3 = cvCreateMat(3,3,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      if ( row == col )
      {
        id3.at<double>(row,col) = 1.0;
      }
      else
      {
        id3.at<double>(row,col) = 0.0;
      }
    }
  }
      
  cv::Mat pcg_crossproduct = cvCreateMat(3,3,CV_64FC1);
  pcg_crossproduct.at<double>(0,0)=0.0;
  pcg_crossproduct.at<double>(0,1)=-(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(0,2)=(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(1,0)=(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(1,1)=0.0;
  pcg_crossproduct.at<double>(1,2)=-(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,0)=-(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(2,1)=(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,2)=0.0;
 
  cv::Mat pcg_mulTransposed = cvCreateMat(pcg.rows, pcg.rows, CV_64FC1);
  cv::mulTransposed (pcg, pcg_mulTransposed, false);
  cv::Mat rcg = ( 1 - cv::norm(pcg) * norm(pcg) /2 ) * id3
    + 0.5 * ( pcg_mulTransposed + sqrt(4 - norm(pcg) * norm(pcg))*pcg_crossproduct);

  //now do the translation
  for ( int i = 0; i < NumberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = SortedMarkerToWorld[i+1].inv() * SortedMarkerToWorld[i];
    mat2 = SortedGridToCamera[i+1] * SortedGridToCamera[i].inv();

    A.at<double>(i*3+0,0)=mat1.at<double>(0,0) - 1.0;
    A.at<double>(i*3+0,1)=mat1.at<double>(0,1) - 0.0;
    A.at<double>(i*3+0,2)=mat1.at<double>(0,2) - 0.0;
    A.at<double>(i*3+1,0)=mat1.at<double>(1,0) - 0.0;
    A.at<double>(i*3+1,1)=mat1.at<double>(1,1) - 1.0;
    A.at<double>(i*3+1,2)=mat1.at<double>(1,2) - 0.0;
    A.at<double>(i*3+2,0)=mat1.at<double>(2,0) - 0.0;
    A.at<double>(i*3+2,1)=mat1.at<double>(2,1) - 0.0;
    A.at<double>(i*3+2,2)=mat1.at<double>(2,2) - 1.0;
 
    cv::Mat m1_t = cvCreateMat(3,1,CV_64FC1);
    cv::Mat m2_t = cvCreateMat(3,1,CV_64FC1);
    for ( int j = 0; j < 3; j ++ )
    {
      m1_t.at<double>(j,0) = mat1.at<double>(j,3);
      m2_t.at<double>(j,0) = mat2.at<double>(j,3);
    }
    cv::Mat b_t = rcg * m2_t - m1_t;
  
    b.at<double>(i*3+0,0)=b_t.at<double>(0,0);
    b.at<double>(i*3+1,0)=b_t.at<double>(1,0);
    b.at<double>(i*3+2,0)=b_t.at<double>(2,0);

  
  }
       
  cv::invert(A,PseudoInverse,CV_SVD);
 
  cv::Mat tcg = PseudoInverse * b;

  Error = A * tcg -b;
 
  cv::mulTransposed (Error, ErrorTransMult, true);
     
  double TransResidual = sqrt(ErrorTransMult.at<double>(0,0)/(NumberOfViews-1));
  residuals[1] = TransResidual;

  cv::Mat CameraToMarker = cvCreateMat(4,4,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      CameraToMarker.at<double>(row,col) = rcg.at<double>(row,col);
    }
  }
  for ( int row = 0; row < 3; row ++ )
  {
    CameraToMarker.at<double>(row,3) = tcg.at<double>(row,0);
  }
  for ( int col = 0; col < 3; col ++ )
  {
    CameraToMarker.at<double>(3,col) = 0.0;
  }
  CameraToMarker.at<double>(3,3)=1.0;
  std::cout << "Camera To Marker Matrix = " << std::endl << CameraToMarker << std::endl;
  std::cout << "Rotational Residual = " << residuals [0] << std::endl;
  std::cout << "Translational Residual = " << residuals [1] << std::endl;

  if ( GroundTruthSolution.length() > 0  )
  {
    std::vector<double> ResultResiduals;
    cv::Mat ResultMatrix = cvCreateMat(4,4,CV_64FC1);
    mitk::LoadResult(GroundTruthSolution, ResultMatrix, ResultResiduals);
    residuals[0] -= ResultResiduals[0];
    residuals[1] -= ResultResiduals[1];
    cv::Scalar Sum = cv::sum(CameraToMarker - ResultMatrix);
    residuals.push_back(Sum[0]);
  }

  return residuals;

}
 
} // end namespace
