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
#include "mitkHandeyeCalibrate.h"
#include <mitkOpenCVFileIOUtils.h>
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <niftkFileHelper.h>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrateFromDirectory::HandeyeCalibrateFromDirectory()
: m_FramesToUse(30)
, m_FramesToUseFactor(3)
, m_SaveProcessedVideoData(true)
, m_VideoInitialised(false)
, m_TrackingDataInitialised(false)
, m_TrackerIndex(0)
, m_AbsTrackerTimingError(20e6) // 20 milliseconds
, m_WriteOutChessboards(true)
, m_WriteOutCalibrationImages(true)
, m_NoVideoSupport(false)
, m_SwapVideoChannels(false)
, m_IntrinsicMatrixLeft(cvCreateMat(3,3,CV_64FC1))
, m_IntrinsicMatrixRight(cvCreateMat(3,3,CV_64FC1))
, m_DistortionCoefficientsLeft(cvCreateMat(1,4,CV_64FC1))
, m_DistortionCoefficientsRight(cvCreateMat(1,4,CV_64FC1))
, m_RotationMatrixRightToLeft(cvCreateMat(3,3,CV_64FC1))
, m_RotationVectorRightToLeft(cvCreateMat(3,1,CV_64FC1))
, m_TranslationVectorRightToLeft(cvCreateMat(3,1,CV_64FC1))
, m_OptimiseIntrinsics(true)
, m_OptimiseRightToLeft(true)
, m_Randomise(false)
{
  m_PixelScaleFactor.Fill(1);
  cvSetIdentity(m_IntrinsicMatrixLeft);
  cvSetIdentity(m_IntrinsicMatrixRight);
  cvSetZero(m_DistortionCoefficientsLeft);
  cvSetZero(m_DistortionCoefficientsRight);
  cvSetIdentity(m_RotationMatrixRightToLeft);
  cvSetZero(m_TranslationVectorRightToLeft);
}


//-----------------------------------------------------------------------------
HandeyeCalibrateFromDirectory::~HandeyeCalibrateFromDirectory()
{
  cvReleaseMat(&m_IntrinsicMatrixLeft);
  cvReleaseMat(&m_IntrinsicMatrixRight);
  cvReleaseMat(&m_DistortionCoefficientsLeft);
  cvReleaseMat(&m_DistortionCoefficientsRight);
  cvReleaseMat(&m_RotationMatrixRightToLeft);
  cvReleaseMat(&m_TranslationVectorRightToLeft);
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::SetOutputDirectory(const std::string& outputDir)
{
  if (outputDir.size() == 0)
  {
    m_OutputDirectory = m_InputDirectory;
  }
  else
  {
    m_OutputDirectory = outputDir;  
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::SetInputDirectory(const std::string& inputDir)
{
  if (inputDir.size() == 0)
  {
    m_InputDirectory = m_OutputDirectory;
  }
  else
  {
    m_InputDirectory = inputDir;  
  }
  this->Modified();  
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::InitialiseOutputDirectory()
{
  if (!niftk::DirectoryExists(m_OutputDirectory)) 
  {
    if (!niftk::CreateDirAndParents(m_OutputDirectory))
    {
      throw std::runtime_error("Failed to create output directory");
    }
  }
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::InitialiseVideo()
{
  std::vector<std::string> filenames = niftk::FindVideoData(m_InputDirectory);
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
void HandeyeCalibrateFromDirectory::InitialiseTracking()
{
  if ( m_Matcher.IsNull() )
  {
    MITK_INFO << "Initialising Video Tracker Matcher";
    m_Matcher = mitk::VideoTrackerMatching::New();
  }
  if ( ! m_Matcher->IsReady() )
  {
    m_Matcher->Initialise (m_InputDirectory);
  }
  
  if ( m_Matcher->IsReady() ) 
  {
    m_TrackingDataInitialised = true;
  }
}
//-----------------------------------------------------------------------------
bool HandeyeCalibrateFromDirectory::LoadExistingIntrinsicCalibrations(std::string directory)
{
  
  cv::Mat litemp = cv::Mat(m_IntrinsicMatrixLeft);
  cv::Mat ldtemp = cv::Mat(m_DistortionCoefficientsLeft);
  cv::Mat ritemp = cv::Mat(m_IntrinsicMatrixRight);
  cv::Mat rdtemp = cv::Mat(m_DistortionCoefficientsRight);

  std::string leftIntrinsicName("calib.left.intrinsic.txt");         
  boost::filesystem::path leftIntrinsicNameFull (directory);
  leftIntrinsicNameFull /= leftIntrinsicName;

  std::string rightIntrinsicName("calib.right.intrinsic.txt");         
  boost::filesystem::path rightIntrinsicNameFull (directory);
  rightIntrinsicNameFull /= rightIntrinsicName;

  mitk::LoadCameraIntrinsicsFromPlainText(leftIntrinsicNameFull.string(), &litemp, &ldtemp);
  mitk::LoadCameraIntrinsicsFromPlainText(rightIntrinsicNameFull.string(), &ritemp, &rdtemp);
  *m_IntrinsicMatrixLeft = CvMat(litemp);
  *m_DistortionCoefficientsLeft = CvMat(ldtemp);
  *m_IntrinsicMatrixRight = CvMat(ritemp);
  *m_DistortionCoefficientsRight = CvMat(rdtemp);
  m_OptimiseIntrinsics=false;
  return true;
}


//-----------------------------------------------------------------------------
bool HandeyeCalibrateFromDirectory::LoadExistingRightToLeft(const std::string& directoryName)
{
  cv::Mat r2lr = cv::Mat(m_RotationMatrixRightToLeft);
  cv::Mat r2lt = cv::Mat(m_TranslationVectorRightToLeft);

  mitk::LoadStereoTransformsFromPlainText(niftk::ConcatenatePath(directoryName, "calib.r2l.txt"), &r2lr, &r2lt);

  *m_RotationMatrixRightToLeft = CvMat(r2lr);
  *m_TranslationVectorRightToLeft = CvMat(r2lt);

  m_OptimiseRightToLeft = false;
  return true;
}


//-----------------------------------------------------------------------------
void HandeyeCalibrateFromDirectory::LoadVideoData(std::string filename)
{
  if ( m_NoVideoSupport ) 
  {
    MITK_WARN << "Ran load video without video support, returning.";
    return;
  }

  cv::VideoCapture *capture;
  try 
  {
    bool ignoreVideoReadFailure = false;
    capture = mitk::InitialiseVideoCapture(filename, ignoreVideoReadFailure) ; 
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    exit(1);
  }
  
  if ( ! capture->isOpened() ) 
  {
    MITK_ERROR << "Failed to open " << filename;
    return;
  }

  if ( ! m_TrackingDataInitialised )
  {
    InitialiseTracking();
  }

  //get frame count doesn't work for 264 files, which are just 
  //raw data get the frame count from the framemap log
  int numberOfFrames = m_Matcher->GetNumberOfFrames();

  double framewidth = capture->get(CV_CAP_PROP_FRAME_WIDTH);
  double frameheight = capture->get(CV_CAP_PROP_FRAME_HEIGHT);
  
  double filesize = numberOfFrames * framewidth * frameheight * 4 / 1e9;
  MITK_INFO << numberOfFrames << "frames in video : " << framewidth << "x" << frameheight;;
  MITK_INFO << filesize << "gigabytes required to store";
  
  //go through data set, randomly selecting frames from a uniform distribution until 
  //sufficient frames are found.
  //First check tracking data is ok (based on timing error threshold)
  //Then try and extract corners in both left and right frames.
  //std::default_random_engine generator;
  //std::uniform_int_distribution<int> distribution (0, numberOfFrames/2);

  if (m_Randomise)
  {
    std::srand(std::time(NULL));
  }
  else
  {
    std::srand(0);
  }

  std::vector <int> leftFramesToCheck;
  std::vector <int> rightFramesToCheck;
  while ( leftFramesToCheck.size() < m_FramesToUse*m_FramesToUseFactor )
  {
    int frameToUse =  std::rand()%(numberOfFrames/2);

    //first check it's not already in array
    if ( (std::find(leftFramesToCheck.begin(), leftFramesToCheck.end(), frameToUse * 2 ) == leftFramesToCheck.end()) )
    {
      MITK_INFO << "Checking tracking for frame pair " << frameToUse * 2 << "," << frameToUse*2 +1;
    
      long long int  leftTimingError;
      long long int  rightTimingError;
      cv::Mat leftTrackingMatrix = m_Matcher->GetTrackerMatrix(frameToUse * 2 ,
        &leftTimingError, m_TrackerIndex );
      cv::Mat rightTrackingMatrix = m_Matcher->GetTrackerMatrix(frameToUse * 2 + 1 ,
        &rightTimingError, m_TrackerIndex );
      if ( std::abs(leftTimingError) > m_AbsTrackerTimingError ||
        std::abs(rightTimingError) > m_AbsTrackerTimingError )
      {
        MITK_INFO << "Rejecting frame " << frameToUse << "Due to high timing error: " <<
          std::abs(leftTimingError) << " > " <<  m_AbsTrackerTimingError;
      }
      else
      {
        //timing error OK, now check if we can extract corners
        leftFramesToCheck.push_back (frameToUse *2);
        rightFramesToCheck.push_back (frameToUse * 2 + 1);
      }
    }
  }

  MITK_INFO << "Got " << leftFramesToCheck.size() << " with accurate enough tracking.";

  //sort the vector in ascending order
  std::sort (leftFramesToCheck.begin(), leftFramesToCheck.end());
  std::sort (rightFramesToCheck.begin(), rightFramesToCheck.end());

  //now go through video once and check if a chessboard can be found.
  int frameNumber = 0 ;
  cv::Size imageSize;
  int numberOfGoodFrames = 0;

  std::vector<cv::Mat>  allLeftImagePoints;
  std::vector<cv::Mat>  allLeftObjectPoints;
  std::vector<cv::Mat>  allRightImagePoints;
  std::vector<cv::Mat>  allRightObjectPoints;
  std::vector<cv::Mat>  allLeftFrames;
  std::vector<cv::Mat>  allRightFrames;
  std::vector<cv::Mat>  allLeftChessBoards;
  std::vector<cv::Mat>  allRightChessBoards;
  std::vector <int>     allLeftFrameNumbers;
  std::vector <int>     allRightFrameNumbers;

  while ( frameNumber < numberOfFrames)
  {
    cv::Mat tempFrame;
    cv::Mat leftFrame;
    cv::Mat leftFrameOrig;
    cv::Mat rightFrame;
    cv::Mat rightFrameOrig;

    *capture >> tempFrame;
    if (!m_SwapVideoChannels)
      leftFrame = tempFrame.clone();
    else
      rightFrame = tempFrame.clone();
    *capture >> tempFrame;
    if (!m_SwapVideoChannels)
      rightFrame = tempFrame.clone();
    else
      leftFrame = tempFrame.clone();

    leftFrameOrig = leftFrame.clone();
    rightFrameOrig = rightFrame.clone();

    imageSize = rightFrame.size();

    if ( (std::find(leftFramesToCheck.begin(), leftFramesToCheck.end(), frameNumber) != leftFramesToCheck.end()) )
    {
      if ((std::find(rightFramesToCheck.begin(), rightFramesToCheck.end(), frameNumber + 1) != rightFramesToCheck.end()) )
      { 
        std::vector <cv::Point2d>* leftImageCorners = new std::vector<cv::Point2d>;
        std::vector <cv::Point3d>* leftObjectCorners = new std::vector<cv::Point3d>;
        bool leftOK = mitk::ExtractChessBoardPoints (
          leftFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres, m_PixelScaleFactor,
          *leftImageCorners, *leftObjectCorners);

        std::vector <cv::Point2d>* rightImageCorners = new std::vector<cv::Point2d>;
        std::vector <cv::Point3d>* rightObjectCorners = new std::vector<cv::Point3d>;
        bool rightOK = mitk::ExtractChessBoardPoints (
          rightFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres, m_PixelScaleFactor,
          *rightImageCorners, *rightObjectCorners);

        if ( leftOK && rightOK )
        {

          MITK_INFO << "Frame " << capture->get(CV_CAP_PROP_POS_FRAMES)-2 << " got " << leftImageCorners->size() << " corners for both images";

          allLeftImagePoints.push_back(cv::Mat(*leftImageCorners,true));
          allLeftObjectPoints.push_back(cv::Mat(*leftObjectCorners,true));
          allRightImagePoints.push_back(cv::Mat(*rightImageCorners,true));
          allRightObjectPoints.push_back(cv::Mat(*rightObjectCorners,true));
          allLeftFrames.push_back(leftFrameOrig);
          allRightFrames.push_back(rightFrameOrig);
          allLeftChessBoards.push_back(leftFrame);
          allRightChessBoards.push_back(rightFrame);
          allLeftFrameNumbers.push_back(frameNumber);
          allRightFrameNumbers.push_back(frameNumber + 1);
          numberOfGoodFrames++;
        }
        else
        {
          MITK_INFO << "Frame " <<  capture->get(CV_CAP_PROP_POS_FRAMES)-2 << " failed corner extraction. Removing from good frame buffer [" << leftImageCorners->size() << "," << rightImageCorners->size() << "].";

          std::vector<int>::iterator newEnd = std::remove(leftFramesToCheck.begin(), leftFramesToCheck.end(), frameNumber);
          leftFramesToCheck.erase(newEnd, leftFramesToCheck.end());
          newEnd = std::remove(rightFramesToCheck.begin(), rightFramesToCheck.end(), frameNumber+1);
          rightFramesToCheck.erase(newEnd, rightFramesToCheck.end());
        }
        
        // buffer contents are copied and stuffed into allLeftImagePoints, etc.
        delete leftImageCorners;
        delete leftObjectCorners;
        delete rightImageCorners;
        delete rightObjectCorners;
      }
      else
      {
        MITK_ERROR << "Left right frame mismatch" ;
        return;
      }
    }
    frameNumber++;
    frameNumber++;
  }

  MITK_INFO << "Got " << allLeftFrameNumbers.size() << " pairs of chessboards.";
  if (allLeftFrameNumbers.size() < m_FramesToUse)
  {
    mitkThrow() << "The number of chessboards (" << allLeftFrameNumbers.size() << ") is less than the number required (" << m_FramesToUse << ")";
  }

  // Now randomly select from the list of available chessboards
  int numberOfChosenFrames = 0;
  std::set<int> setOfChosenIndexes;
  std::vector <int> leftFramesToUse;
  std::vector <int> rightFramesToUse;
  std::vector<cv::Mat> chosenLeftImagePoints;
  std::vector<cv::Mat> chosenLeftObjectPoints;
  std::vector<cv::Mat> chosenRightImagePoints;
  std::vector<cv::Mat> chosenRightObjectPoints;

  while (numberOfChosenFrames < m_FramesToUse)
  {
    int indexToUse = std::rand() % allLeftFrameNumbers.size();
    if (setOfChosenIndexes.find(indexToUse) == setOfChosenIndexes.end())
    {
      setOfChosenIndexes.insert(indexToUse);
      numberOfChosenFrames++;

      MITK_INFO << "Selecting image " << numberOfChosenFrames << " as frame " << allLeftFrameNumbers[indexToUse] << ", " << allRightFrameNumbers[indexToUse];

      // We have randomly selected from the list of available chessboards.
      // So, copy to the buffers that will be required for calibration.
      leftFramesToUse.push_back(allLeftFrameNumbers[indexToUse]);
      rightFramesToUse.push_back(allRightFrameNumbers[indexToUse]);
      chosenLeftImagePoints.push_back(allLeftImagePoints[indexToUse]);
      chosenLeftObjectPoints.push_back(allLeftObjectPoints[indexToUse]);
      chosenRightImagePoints.push_back(allRightImagePoints[indexToUse]);
      chosenRightObjectPoints.push_back(allRightObjectPoints[indexToUse]);

      if ( m_WriteOutCalibrationImages )
      {
        std::string leftFilename = m_OutputDirectory + "/LeftFrame" + boost::lexical_cast<std::string>(numberOfChosenFrames) + ".png";
        MITK_INFO << "Writing image to " << leftFilename;
        cv::imwrite( leftFilename, allLeftFrames[indexToUse] );

        std::string rightFilename = m_OutputDirectory + "/RightFrame" + boost::lexical_cast<std::string>(numberOfChosenFrames) + ".png";
        MITK_INFO << "Writing image to " << rightFilename;
        cv::imwrite( rightFilename, allRightFrames[indexToUse] );
      }
      if ( m_WriteOutChessboards )
      {
        std::string leftFilename = m_OutputDirectory + "/LeftChessboard" + boost::lexical_cast<std::string>(leftFramesToUse[indexToUse]) + ".png";
        MITK_INFO << "Writing image to " << leftFilename;
        cv::imwrite( leftFilename, allLeftChessBoards[indexToUse] );

        std::string rightFilename = m_OutputDirectory + "/RightChessboard" + boost::lexical_cast<std::string>(rightFramesToUse[indexToUse]) + ".png";
        MITK_INFO << "Writing image to " << rightFilename;
        cv::imwrite( rightFilename, allRightChessBoards[indexToUse] );
      }
    }
  }

  MITK_INFO << "There are " << leftFramesToUse.size() << " chosen frames.";
  if (leftFramesToUse.size() < m_FramesToUse)
  {
    mitkThrow() << "Chose " << leftFramesToUse.size() << ", instead of " << m_FramesToUse;
  }

  cv::Mat leftImagePoints (m_NumberCornersWidth * m_NumberCornersHeight * leftFramesToUse.size(),2,CV_64FC1);
  cv::Mat leftObjectPoints (m_NumberCornersWidth * m_NumberCornersHeight * leftFramesToUse.size(),3,CV_64FC1);
  cv::Mat rightImagePoints (m_NumberCornersWidth * m_NumberCornersHeight * leftFramesToUse.size(),2,CV_64FC1);
  cv::Mat rightObjectPoints (m_NumberCornersWidth * m_NumberCornersHeight * leftFramesToUse.size(),3,CV_64FC1);
  
  cv::Mat leftPointCounts (leftFramesToUse.size(),1,CV_32SC1);
  cv::Mat rightPointCounts (leftFramesToUse.size(),1,CV_32SC1);

  if  (chosenLeftImagePoints.size()   !=  leftFramesToUse.size() ||
       chosenLeftObjectPoints.size()  !=  leftFramesToUse.size() ||
       chosenRightImagePoints.size()  !=  leftFramesToUse.size() ||
       chosenRightObjectPoints.size() !=  leftFramesToUse.size() )
  {
    mitkThrow() << "Detected unequal matrix sizes";
  }

  for ( unsigned int i = 0 ; i < leftFramesToUse.size() ; i ++  )
  {
    unsigned int size1 = chosenLeftImagePoints[i].size().height;
    //FIX ME
    unsigned int size2 = chosenLeftObjectPoints[i].size().height;
    unsigned int size3 = chosenRightImagePoints[i].size().height;
    unsigned int size4 = chosenRightObjectPoints[i].size().height;
  
    if (size1 != m_NumberCornersWidth * m_NumberCornersHeight ||
        size2 != m_NumberCornersWidth * m_NumberCornersHeight ||
        size3 != m_NumberCornersWidth * m_NumberCornersHeight ||
        size4 != m_NumberCornersWidth * m_NumberCornersHeight)
    {
      mitkThrow() << "Detected unequal matrix sizes";
    }
  }

  for ( unsigned int i = 0 ; i < leftFramesToUse.size() ; i++ )
  {
    MITK_INFO << "Filling "  << i;
    for ( unsigned int j = 0 ; j < m_NumberCornersWidth * m_NumberCornersHeight ; j ++ ) 
    {
      leftImagePoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        chosenLeftImagePoints[i].at<double>(j,0);
      leftImagePoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        chosenLeftImagePoints[i].at<double>(j,1);
     
      leftObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        chosenLeftObjectPoints[i].at<double>(j,0);
      leftObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        chosenLeftObjectPoints[i].at<double>(j,1);
      leftObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,2) =
        chosenLeftObjectPoints[i].at<double>(j,2);

      rightImagePoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        chosenRightImagePoints[i].at<double>(j,0);
      rightImagePoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        chosenRightImagePoints[i].at<double>(j,1);
      
      rightObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,0) =
        chosenRightObjectPoints[i].at<double>(j,0);
      rightObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,1) =
        chosenRightObjectPoints[i].at<double>(j,1);
      rightObjectPoints.at<double>(i* m_NumberCornersWidth * m_NumberCornersHeight + j,2) =
        chosenRightObjectPoints[i].at<double>(j,2);

    }
    leftPointCounts.at<int>(i,0) = m_NumberCornersWidth * m_NumberCornersHeight;
    rightPointCounts.at<int>(i,0) = m_NumberCornersWidth * m_NumberCornersHeight;
  }
  if ( true )
  {
    std::string leftimagePointsfilename = m_OutputDirectory + "/LeftImagePoints.xml";
    std::string leftObjectPointsfilename = m_OutputDirectory + "/LeftObjectPoints.xml";
    std::string rightimagePointsfilename = m_OutputDirectory + "/RightImagePoints.xml";
    std::string rightObjectPointsfilename = m_OutputDirectory + "/RightObjectPoints.xml";
    std::string leftPointCountfilename = m_OutputDirectory + "/LeftPointCount.xml";
    std::string rightPointCountfilename = m_OutputDirectory + "/RightPointCount.xml";
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
  CvMat* outputRotationVectorsLeft = cvCreateMat(leftFramesToUse.size(),3,CV_64FC1);
  CvMat* outputTranslationVectorsLeft= cvCreateMat(leftFramesToUse.size(),3,CV_64FC1);
  CvMat* outputRotationVectorsRight= cvCreateMat(leftFramesToUse.size(),3,CV_64FC1);
  CvMat* outputTranslationVectorsRight= cvCreateMat(leftFramesToUse.size(),3,CV_64FC1);
  CvMat* outputEssentialMatrix = cvCreateMat(3,3,CV_64FC1);
  CvMat* outputFundamentalMatrix= cvCreateMat(3,3,CV_64FC1);

  double reprojectionError = mitk::CalibrateStereoCameraParameters(
      leftObjectPoints,
      leftImagePoints,
      leftPointCounts,
      imageSize,
      rightObjectPoints,
      rightImagePoints,
      rightPointCounts,
      *m_IntrinsicMatrixLeft,
      *m_DistortionCoefficientsLeft,
      *outputRotationVectorsLeft,
      *outputTranslationVectorsLeft,
      *m_IntrinsicMatrixRight,
      *m_DistortionCoefficientsRight,
      *outputRotationVectorsRight,
      *outputTranslationVectorsRight,
      *m_RotationMatrixRightToLeft,
      *m_TranslationVectorRightToLeft,
      *outputEssentialMatrix,
      *outputFundamentalMatrix,
      ! m_OptimiseIntrinsics,
      ! m_OptimiseRightToLeft
      );
  
  //write it out
  std::string leftIntrinsic = m_OutputDirectory + "/calib.left.intrinsic.txt";
  std::string rightIntrinsic = m_OutputDirectory + "/calib.right.intrinsic.txt";
  std::string rightToLeft = m_OutputDirectory + "/calib.r2l.txt";
  std::string extrinsic = m_OutputDirectory + "/leftextrinsics.txt";

  int outputPrecision = 10;
  int outputWidth = 10;

  std::ofstream fs_leftIntrinsic;
  fs_leftIntrinsic.precision(outputPrecision);
  fs_leftIntrinsic.width(outputWidth);

  std::ofstream fs_rightIntrinsic;
  fs_rightIntrinsic.precision(outputPrecision);
  fs_rightIntrinsic.width(outputWidth);

  std::ofstream fs_r2l;
  fs_r2l.precision(outputPrecision);
  fs_r2l.width(outputWidth);

  std::ofstream fs_ext;
  fs_ext.precision(outputPrecision);
  fs_ext.width(outputWidth);

  fs_leftIntrinsic.open(leftIntrinsic.c_str(), std::ios::out);
  fs_rightIntrinsic.open(rightIntrinsic.c_str(), std::ios::out);
  fs_r2l.open(rightToLeft.c_str(), std::ios::out);
  fs_ext.open(extrinsic.c_str(), std::ios::out);

  for ( int row = 0 ; row < 3 ; row ++ ) 
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      fs_leftIntrinsic << CV_MAT_ELEM (*m_IntrinsicMatrixLeft, double, row,col) << " ";
      fs_rightIntrinsic << CV_MAT_ELEM (*m_IntrinsicMatrixRight, double, row,col) << " ";
      fs_r2l << CV_MAT_ELEM (*m_RotationMatrixRightToLeft, double , row,col) << " ";
    }
    fs_leftIntrinsic << std::endl;
    fs_rightIntrinsic << std::endl;
    fs_r2l << std::endl;
  }
  for ( int i = 0 ; i < 4 ; i ++ )  
  {
    fs_leftIntrinsic << CV_MAT_ELEM (*m_DistortionCoefficientsLeft, double , 0, i ) << " ";
    fs_rightIntrinsic << CV_MAT_ELEM (*m_DistortionCoefficientsRight, double , 0, i ) << " ";
  }
  fs_leftIntrinsic << std::endl;
  fs_rightIntrinsic << std::endl;

  for ( int i = 0 ; i < 3 ; i ++ )  
  {
    fs_r2l << CV_MAT_ELEM (*m_TranslationVectorRightToLeft, double , i, 0 ) << " ";
  }
  fs_r2l << std::endl;

  fs_leftIntrinsic.close();
  fs_rightIntrinsic.close();
  fs_r2l.close();

  std::string trackerDirectory = m_OutputDirectory + "/TrackerMatrices" + boost::lexical_cast<std::string>(m_TrackerIndex);
  niftk::CreateDirAndParents(trackerDirectory);

  for ( unsigned int view = 0 ; view < leftFramesToUse.size() ; view ++ )
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

    cv::Mat LeftTrackingMatrix = m_Matcher->GetTrackerMatrix(leftFramesToUse[view], NULL, m_TrackerIndex );

    std::string trackerFilename = trackerDirectory + "/" + boost::lexical_cast<std::string>(view) + ".txt";
    MITK_INFO << "Saving matrix for frame " << leftFramesToUse[view] << "to " << trackerFilename;
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
  
  m_VideoInitialised = true;

  Calibrate ( m_OutputDirectory + "/TrackerMatrices" + boost::lexical_cast<std::string>(m_TrackerIndex) , extrinsic); 

  cv::Mat handEyeRotationMatrix(m_CameraToMarker, cv::Range(0, 2), cv::Range(0,2));
  cv::Mat handEyeRotationVector(cvCreateMat(3,1,CV_64FC1));
  cv::Rodrigues(handEyeRotationMatrix, handEyeRotationVector);

  // for spreadsheet/analysis purposes. A big ugly, but amenable to grepping through log files.
  MITK_INFO << "Calibration Summary:"
            << numberOfGoodFrames << ", "
            << reprojectionError << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixLeft, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixLeft, double, 1,1) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixLeft, double, 0,2) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixLeft, double, 1,2) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsLeft, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsLeft, double, 0,1) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsLeft, double, 0,2) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsLeft, double, 0,3) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixRight, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixRight, double, 1,1) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixRight, double, 0,2) << ", "
            << CV_MAT_ELEM (*m_IntrinsicMatrixRight, double, 1,2) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsRight, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsRight, double, 0,1) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsRight, double, 0,2) << ", "
            << CV_MAT_ELEM (*m_DistortionCoefficientsRight, double, 0,3) << ", "
            << CV_MAT_ELEM (*m_RotationVectorRightToLeft, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_RotationVectorRightToLeft, double, 1,0) << ", "
            << CV_MAT_ELEM (*m_RotationVectorRightToLeft, double, 2,0) << ", "
            << CV_MAT_ELEM (*m_TranslationVectorRightToLeft, double, 0,0) << ", "
            << CV_MAT_ELEM (*m_TranslationVectorRightToLeft, double, 1,0) << ", "
            << CV_MAT_ELEM (*m_TranslationVectorRightToLeft, double, 2,0) << ", "
            << handEyeRotationVector.at<double>(0, 0) << ", "
            << handEyeRotationVector.at<double>(1, 0) << ", "
            << handEyeRotationVector.at<double>(2, 0) << ", "
            << m_CameraToMarker.at<double>(0,3) << ", "
            << m_CameraToMarker.at<double>(1,3) << ", "
            << m_CameraToMarker.at<double>(2,3);
}
 
} // end namespace
