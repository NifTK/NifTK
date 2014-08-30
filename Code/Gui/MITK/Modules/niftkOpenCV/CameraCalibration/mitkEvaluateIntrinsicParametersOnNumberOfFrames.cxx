/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkEvaluateIntrinsicParametersOnNumberOfFrames.h"
#include "mitkOpenCVFileIOUtils.h"
#include "niftkFileHelper.h"
#include "mitkHandeyeCalibrateFromDirectory.h"
#include "mitkCameraCalibrationFacade.h"
#include "mitkHandeyeCalibrate.h"
#include <mitkExceptionMacro.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <opencv2/opencv.hpp>
#include <highgui.h>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
EvaluateIntrinsicParametersOnNumberOfFrames::EvaluateIntrinsicParametersOnNumberOfFrames()
: m_FramesToUse(40)
, m_VideoInitialised(false)
, m_TrackingDataInitialised(false)
, m_AbsTrackerTimingError(20e6) // 20 milliseconds
, m_NumberCornersWidth(14)
, m_NumberCornersHeight(10)
, m_SquareSizeInMillimetres(3.0)
, m_IntrinsicMatrixLeft(cvCreateMat(3,3,CV_64FC1))
, m_IntrinsicMatrixRight(cvCreateMat(3,3,CV_64FC1))
, m_DistortionCoefficientsLeft(cvCreateMat(1,4,CV_64FC1))
, m_DistortionCoefficientsRight(cvCreateMat(1,4,CV_64FC1))
, m_OptimiseIntrinsics(true)
, m_PostProcessExtrinsicsAndR2L(false)
, m_PostProcessR2LThenExtrinsics(false)
, m_SwapVideoChannels(false)
, m_NumberOfFrames(0)
{
  m_PixelScaleFactor.Fill(1);
}


//-----------------------------------------------------------------------------
EvaluateIntrinsicParametersOnNumberOfFrames::~EvaluateIntrinsicParametersOnNumberOfFrames()
{
  m_TimeStamps.clear();
  m_MatchedVideoFrames.clear();
  m_MatchedTrackingMatrix.clear();
  cvReleaseMat(&m_IntrinsicMatrixLeft);
  cvReleaseMat(&m_IntrinsicMatrixRight);
  cvReleaseMat(&m_DistortionCoefficientsLeft);
  cvReleaseMat(&m_DistortionCoefficientsRight);
}

//-----------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::InitialiseOutputDirectory()
{
  if (!niftk::DirectoryExists(m_OutputDirectory)) 
  {
    if (!niftk::CreateDirAndParents(m_OutputDirectory))
    {
      throw std::runtime_error("Failed to create output directory");
    }
  }
	
	// create sub-dir to save the detected left and right image corners
	m_LeftDirectory = m_OutputDirectory + "\\Left";
  niftk::CreateDirAndParents(m_LeftDirectory);
  niftk::CreateDirAndParents(m_LeftDirectory + "\\img");
  niftk::CreateDirAndParents(m_LeftDirectory + "\\obj");
	
  m_RightDirectory = m_OutputDirectory + "\\Right";
  niftk::CreateDirAndParents(m_RightDirectory);
  niftk::CreateDirAndParents(m_RightDirectory + "\\img");
  niftk::CreateDirAndParents(m_RightDirectory + "\\obj");
}

//-----------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::InitialiseVideo()
{
  std::vector<std::string> filenames = niftk::FindVideoData(m_InputDirectory);
  if ( filenames.size() == 0 ) 
  {
    MITK_ERROR << "Failed to find any video files";
    return;
  }
  if ( filenames.size() > 1 ) 
  {
    MITK_ERROR << "Found too many video files. ";
    for ( unsigned int  i = 0 ; i < filenames.size() ; i++ )
    {
      MITK_ERROR << filenames[i];
    }
    return;
  }

  this->ReadTimeStampFromLogFile();
	this->MatchingVideoFramesToTrackingMatrix();

  MITK_INFO << "Loading video frames from " << filenames[0];
  this->LoadVideoData (filenames[0]);
  return;
      
}

//-----------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::ReadTimeStampFromLogFile()
{
  std::vector<std::string> frampmapfiles = mitk::FindVideoFrameMapFiles(m_InputDirectory);

  std::ifstream fin(frampmapfiles[0].c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open frame map file " << frampmapfiles[0].c_str();
    return;
  }

  std::string line;
  unsigned int frameNumber; 
  unsigned int sequenceNumber;
  unsigned int channel;
  unsigned long long timeStamp;
  unsigned int linenumber = 0;

  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      bool parseSuccess = linestream >> frameNumber >> sequenceNumber >> channel >> timeStamp;

      m_TimeStamps.push_back(timeStamp); 
      m_NumberOfFrames++;
    }
  }
}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::MatchingVideoFramesToTrackingMatrix()
{     
  TrackingMatrixTimeStamps mtxTimeStamps = mitk::FindTrackingTimeStamps(m_InputMatrixDirectory);

  long long timingError;
  unsigned long long targetTimeStamp; 
	
	m_MatchedTrackingMatrix.clear();
	m_MatchedVideoFrames.clear();

  unsigned int size = m_TimeStamps.size();

  for ( unsigned int i=0; i<size; i=i+2 )
  {
    targetTimeStamp = mtxTimeStamps.GetNearestTimeStamp(m_TimeStamps[i], &timingError);

    if ( timingError < m_AbsTrackerTimingError && timingError > -m_AbsTrackerTimingError )
    {
		  m_MatchedVideoFrames.push_back(m_TimeStamps[i]);
			m_MatchedTrackingMatrix.push_back(targetTimeStamp);
    }
  }
}

//-----------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::LoadVideoData(std::string filename)
{
  bool ignoreVideoReadFailure = false;
  cv::VideoCapture *capture;
  try 
  {
    capture = mitk::InitialiseVideoCapture(filename) ; 
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught exception:" << e.what();
    exit(1);         
  }
  
  if ( ! capture->isOpened() ) 
  {
    MITK_ERROR << "Failed to open " << filename;
    return;
  }

  //now go through video and extract frames to use

  unsigned int size = m_TimeStamps.size();

  for ( unsigned int nframe=0; nframe<size; nframe=nframe+2 )
  {
    cv::Mat tempFrame;
    cv::Mat leftFrame;
    cv::Mat rightFrame;
		
		bool find = false;
		
		for ( int i=0; i<m_MatchedVideoFrames.size(); i++ )
		{
		  if ( m_TimeStamps[nframe] == m_MatchedVideoFrames[i] )
			{
			  find = true;
				break;
			}
		}
		
		if ( !find )
    {
		  *capture >> tempFrame;
			*capture >> tempFrame;
			continue;
    }		

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
			
	  std::vector <cv::Point2d>* leftImageCorners = new std::vector<cv::Point2d>;
    std::vector <cv::Point3d>* leftObjectCorners = new std::vector<cv::Point3d>;
    std::vector <cv::Point2d>* rightImageCorners = new std::vector<cv::Point2d>;
    std::vector <cv::Point3d>* rightObjectCorners = new std::vector<cv::Point3d>;
    bool LeftOK = false;
    bool RightOK = false;
    
    LeftOK = mitk::ExtractChessBoardPoints (
          leftFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres, m_PixelScaleFactor,
          *leftImageCorners, *leftObjectCorners);

    if ( LeftOK && (leftImageCorners->size()==m_NumberCornersWidth*m_NumberCornersHeight) )
    {
      RightOK = mitk::ExtractChessBoardPoints (
          rightFrame, m_NumberCornersWidth,
          m_NumberCornersHeight, 
          true, m_SquareSizeInMillimetres, m_PixelScaleFactor,
          *rightImageCorners, *rightObjectCorners);
    }
		
		// save the detected corners into .txt file
		if ( LeftOK && RightOK 
		  && (leftImageCorners->size()==m_NumberCornersWidth*m_NumberCornersHeight) 
			&& (rightImageCorners->size()==m_NumberCornersWidth*m_NumberCornersHeight) )
		{
			std::string tName = boost::lexical_cast<std::string>(m_MatchedVideoFrames[nframe/2]) + ".txt";
      std::string  leftimgNameFull = m_LeftDirectory + "\\img\\" + tName;
			std::string  rightimgNameFull = m_RightDirectory + "\\img\\" + tName;
			
      std::string  leftobjNameFull = m_LeftDirectory + "\\obj\\" + tName;
			std::string  rightobjNameFull = m_RightDirectory + "\\obj\\" + tName;
			
			std::ofstream fimgl( leftimgNameFull.c_str() );
			std::ofstream fobjl( leftobjNameFull.c_str() );
			
		  std::ofstream fimgr( rightimgNameFull.c_str() );
			std::ofstream fobjr( rightobjNameFull.c_str() );
			
	    if( !fimgl.is_open() || !fimgr.is_open() || !fobjl.is_open()|| !fobjr.is_open() )
	    {
		    std::cerr << "Error opening file" << std::endl ;
	    }
			else
			{
		    for ( int i=0; i<leftImageCorners->size(); i++ )
			  {
			    fimgl << (leftImageCorners->at(i)).x << " " << (leftImageCorners->at(i)).y << std::endl; 
			  }
			  for ( int i=0; i<leftObjectCorners->size(); i++ )
			  {
			    fobjl << (leftObjectCorners->at(i)).x << " " << (leftObjectCorners->at(i)).y << " " << (leftObjectCorners->at(i)).z << std::endl; 
			  }
				
				for ( int i=0; i<rightImageCorners->size(); i++ )
			  {
			    fimgr << (rightImageCorners->at(i)).x << " " << (rightImageCorners->at(i)).y << std::endl; 
			  }
			  for ( int i=0; i<rightObjectCorners->size(); i++ )
			  {
			    fobjr << (rightObjectCorners->at(i)).x << " " << (rightObjectCorners->at(i)).y << " " << (rightObjectCorners->at(i)).z << std::endl; 
			  }
				
			  fimgl.close();
			  fobjl.close();
				fimgr.close();
			  fobjr.close();
			}
		}

    leftImageCorners->clear();
    leftObjectCorners->clear();
    rightImageCorners->clear();
    rightObjectCorners->clear();
    delete leftImageCorners;
    delete leftObjectCorners;
    delete rightImageCorners;
    delete rightObjectCorners;
  }
}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::RunExperiment()
{
  // Get frame size.
  std::vector<std::string> filenames = niftk::FindVideoData(m_InputDirectory);
  if ( filenames.size() == 0 ) 
  {
    MITK_ERROR << "Failed to find any video files";
    return;
  }
  if ( filenames.size() > 1 ) 
  {
    MITK_ERROR << "Found too many video files. ";
    for ( unsigned int  i = 0 ; i < filenames.size() ; i++ )
    {
      MITK_ERROR << filenames[i];
    }
    return;
  }

  cv::VideoCapture *capture;
  try 
  {
    capture = mitk::InitialiseVideoCapture(filenames[0]) ; 
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught exception:" << e.what();
    exit(1); 
  }

  if ( ! capture->isOpened() ) 
  {
    MITK_ERROR << "Failed to open " << filenames[0];
    return;
  }
  
  double framewidth = capture->get(CV_CAP_PROP_FRAME_WIDTH);
  double frameheight = capture->get(CV_CAP_PROP_FRAME_HEIGHT);

  cv::Size imageSize(framewidth, frameheight);

  int outputPrecision = 3;
  int repetition = 20;

  // Compute the intrinsic parameters and output to files.
  std::string leftName = m_OutputDirectory + "\\calib.left." + boost::lexical_cast<std::string>(m_FramesToUse) + ".txt";
  std::string rightName = m_OutputDirectory + "\\calib.right." + boost::lexical_cast<std::string>(m_FramesToUse) + ".txt";

  std::ofstream fs_left;
  fs_left.precision(outputPrecision);

  std::ofstream fs_right;
  fs_right.precision(outputPrecision);

  fs_left.open(leftName.c_str(), std::ios::out);
  fs_right.open(rightName.c_str(), std::ios::out);

  for ( int i=0; i<repetition; i++ )
  {
    this->ComputeCamaraIntrinsicParameters(imageSize);

    fs_left << cv::Mat(m_IntrinsicMatrixLeft) << '\n';
    fs_left << cv::Mat(m_DistortionCoefficientsLeft) << '\n' << '\n';

    fs_right << cv::Mat(m_IntrinsicMatrixRight) << '\n';
    fs_right << cv::Mat(m_DistortionCoefficientsRight) << '\n' << '\n';
  }

  fs_left.close();
  fs_right.close();

}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::ComputeCamaraIntrinsicParameters(cv::Size &imageSize)
{
  std::vector<std::string> fnvec_limg;
  std::vector<std::string> fnvec_rimg; 
  std::vector<std::string> fnvec_lobj; 
  std::vector<std::string> fnvec_robj; 
  std::vector<std::string> fnvec_mtx;
  this->PairewiseFiles( fnvec_limg, fnvec_rimg, fnvec_lobj, fnvec_robj, fnvec_mtx);

  int sz = fnvec_limg.size();

  //generate random m_FramesToUse frames
  srand (time(NULL));
  std::vector<int> frametouse;

  int counter = 0;
  while ( counter<m_FramesToUse )
  {
    int num = rand() % sz;

    if ( (std::find(frametouse.begin(), frametouse.end(), num ) == frametouse.end()) )
    {
      frametouse.push_back(num);
      counter++;
    }
  }

  int nCorners = m_NumberCornersWidth * m_NumberCornersHeight;

  cv::Mat leftImagePoints (nCorners * m_FramesToUse,2,CV_64FC1);
  cv::Mat leftObjectPoints (nCorners * m_FramesToUse,3,CV_64FC1);
  cv::Mat rightImagePoints (nCorners * m_FramesToUse,2,CV_64FC1);
  cv::Mat rightObjectPoints (nCorners * m_FramesToUse,3,CV_64FC1);

  cv::Mat leftPointCounts (m_FramesToUse,1,CV_32SC1);
  cv::Mat rightPointCounts (m_FramesToUse,1,CV_32SC1);

  for ( int i=0; i<m_FramesToUse; i++ )
  {
    std::vector<float*> lipts;
    std::vector<float*> ripts;
    std::vector<float*> lopts;
    std::vector<float*> ropts;

    if ( this->Read2DPointsFromFile(fnvec_limg[frametouse[i]].c_str(),lipts) != nCorners )
    {
      return;
    }
    
    if ( this->Read2DPointsFromFile(fnvec_rimg[frametouse[i]].c_str(),ripts) != nCorners )
    {
      return;
    }

    if ( this->Read3DPointsFromFile(fnvec_lobj[frametouse[i]].c_str(),lopts) != nCorners )
    {
      return;
    }
    
    if ( this->Read3DPointsFromFile(fnvec_robj[frametouse[i]].c_str(),ropts) != nCorners )
    {
      return;
    }

    for ( int j = 0 ; j < nCorners ; j++ )
    {
      leftImagePoints.at<double>(i* nCorners + j,0) = lipts[j][0];
      leftImagePoints.at<double>(i* nCorners + j,1) = lipts[j][1];

      rightImagePoints.at<double>(i*nCorners + j,0) = ripts[j][0];
      rightImagePoints.at<double>(i* nCorners + j,1) = ripts[j][1];
    }

    for ( int j = 0 ; j < nCorners ; j++ )
    {
      leftObjectPoints.at<double>(i* nCorners + j,0) = lopts[j][0];
      leftObjectPoints.at<double>(i* nCorners + j,1) = lopts[j][1];
      leftObjectPoints.at<double>(i* nCorners + j,2) = lopts[j][2];

      rightObjectPoints.at<double>(i* nCorners + j,0) = ropts[j][0];
      rightObjectPoints.at<double>(i* nCorners + j,1) = ropts[j][1];
      rightObjectPoints.at<double>(i* nCorners + j,2) = ropts[j][2];
    }

    leftPointCounts.at<int>(i,0) = nCorners;
    rightPointCounts.at<int>(i,0) = nCorners;
  }

  if ( false )
  {
    std::string leftimagePointsfilename = m_OutputDirectory + "\\LeftImagePoints.xml";
    std::string leftObjectPointsfilename = m_OutputDirectory + "\\LeftObjectPoints.xml";
    std::string rightimagePointsfilename = m_OutputDirectory + "\\RightImagePoints.xml";
    std::string rightObjectPointsfilename = m_OutputDirectory + "\\RightObjectPoints.xml";
    std::string leftPointCountfilename = m_OutputDirectory + "\\LeftPointCount.xml";
    std::string rightPointCountfilename = m_OutputDirectory + "\\RightPointCount.xml";
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

  std::cout << "Starting intrinisic calibration";

  CvMat* outputRotationVectorsLeft = cvCreateMat(m_FramesToUse,3,CV_64FC1);
  CvMat* outputTranslationVectorsLeft= cvCreateMat(m_FramesToUse,3,CV_64FC1);
  CvMat* outputRotationVectorsRight= cvCreateMat(m_FramesToUse,3,CV_64FC1);
  CvMat* outputTranslationVectorsRight= cvCreateMat(m_FramesToUse,3,CV_64FC1);
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
      //m_PostProcessExtrinsicsAndR2L,
      //m_PostProcessR2LThenExtrinsics,
      *m_IntrinsicMatrixLeft,
      *m_DistortionCoefficientsLeft,
      *outputRotationVectorsLeft,
      *outputTranslationVectorsLeft,
      *m_IntrinsicMatrixRight,
      *m_DistortionCoefficientsRight,
      *outputRotationVectorsRight,
      *outputTranslationVectorsRight,
      *outputRightToLeftRotation,
      *outputRightToLeftTranslation,
      *outputEssentialMatrix,
      *outputFundamentalMatrix,
      ! m_OptimiseIntrinsics
);

  std::cout << "IntrinsicMatrixLeft : " << std::endl << cv::Mat(m_IntrinsicMatrixLeft) << std::endl;
  std::cout << "IntrinsicMatrixRight : " << std::endl << cv::Mat(m_IntrinsicMatrixRight) << std::endl;

  cvReleaseMat(&outputRotationVectorsLeft);
  cvReleaseMat(&outputTranslationVectorsLeft);
  cvReleaseMat(&outputRotationVectorsRight);
  cvReleaseMat(&outputTranslationVectorsRight);
  cvReleaseMat(&outputRightToLeftRotation);
  cvReleaseMat(&outputRightToLeftTranslation);
  cvReleaseMat(&outputEssentialMatrix);
  cvReleaseMat(&outputFundamentalMatrix);
}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::PairewiseFiles(std::vector<std::string> &fnvec_limg, 
                                                                 std::vector<std::string> &fnvec_rimg, 
                                                                 std::vector<std::string> &fnvec_lobj, 
                                                                 std::vector<std::string> &fnvec_robj, 
                                                                 std::vector<std::string> &fnvec_mtx)
{     
  TrackingMatrixTimeStamps mtxTimeStamps = mitk::FindTrackingTimeStamps(m_InputMatrixDirectory);
  TrackingMatrixTimeStamps imgStamps = mitk::FindTrackingTimeStamps(m_LeftDirectory + "\\img" );

  m_MatchedVideoFrames.clear();
  m_MatchedTrackingMatrix.clear();

  long long timingError;
  unsigned long long targetTimeStamp; 

  int size;
  if ( mtxTimeStamps.GetSize() < imgStamps.GetSize() )
  {
    size = mtxTimeStamps.GetSize();

    for ( int i=0; i<size; i++ )
    {
      targetTimeStamp = imgStamps.GetNearestTimeStamp(mtxTimeStamps.GetTimeStamp(i), &timingError);

      if ( timingError < m_AbsTrackerTimingError && timingError > -m_AbsTrackerTimingError )
      {
        std::string matrixFileName = boost::lexical_cast<std::string>(mtxTimeStamps.GetTimeStamp(i)) + ".txt";
        std::string  matrixFileNameFull = m_InputMatrixDirectory + "\\" + matrixFileName;
        fnvec_mtx.push_back(matrixFileNameFull);

        m_MatchedTrackingMatrix.push_back(mtxTimeStamps.GetTimeStamp(i));
        m_MatchedVideoFrames.push_back(targetTimeStamp);
      }
    }
  }
  else
  {
    size = imgStamps.GetSize();

    for ( int i=0; i<size; i++ )
    {
      targetTimeStamp = mtxTimeStamps.GetNearestTimeStamp(imgStamps.GetTimeStamp(i), &timingError);

      if ( timingError < m_AbsTrackerTimingError && timingError > -m_AbsTrackerTimingError )
      {
        std::string matrixFileName = boost::lexical_cast<std::string>(targetTimeStamp) + ".txt";
        std::string  matrixFileNameFull = m_InputMatrixDirectory + "\\" + matrixFileName;
        fnvec_mtx.push_back(matrixFileNameFull);

        m_MatchedTrackingMatrix.push_back(targetTimeStamp);
        m_MatchedVideoFrames.push_back(imgStamps.GetTimeStamp(i));
      }
    } 
  }

  // create corner file names.
  for ( int i=0; i<m_MatchedVideoFrames.size(); i++ )
  {
    std::string tName = boost::lexical_cast<std::string>(m_MatchedVideoFrames[i]) + ".txt";

    std::string  limgNameFull = m_LeftDirectory + "\\img\\" + tName;
    fnvec_limg.push_back(limgNameFull);

    std::string  lobjNameFull = m_LeftDirectory + "\\obj\\" + tName;
    fnvec_lobj.push_back(lobjNameFull);

    std::string  rimgNameFull = m_RightDirectory + "\\img\\" + tName;
    fnvec_rimg.push_back(rimgNameFull);

    std::string  robjNameFull = m_RightDirectory + "\\obj\\" + tName;
    fnvec_robj.push_back(robjNameFull);
  }
}

//---------------------------------------------------------------------------
int EvaluateIntrinsicParametersOnNumberOfFrames::Read2DPointsFromFile(const char *fn, std::vector<float*> &ptvec)
{
  std::ifstream fin(fn);
  if ( !fin )
  {
    std::cout << "Failed to open file " << fn;
    return 0;
  }

  std::string line;
  int counter = 0;

  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      float *pt = new float[2]; 
      bool parseSuccess = linestream >> pt[0] >> pt[1];

      ptvec.push_back(pt); 
      counter++;
    }
  }
  return counter;
}

//---------------------------------------------------------------------------
int EvaluateIntrinsicParametersOnNumberOfFrames::Read3DPointsFromFile(const char *fn, std::vector<float*> &ptvec)
{
  std::ifstream fin(fn);
  if ( !fin )
  {
    std::cout << "Failed to open file " << fn;
    return 0;
  }

  std::string line;
  int counter = 0;

  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      float *pt = new float[3]; 
      bool parseSuccess = linestream >> pt[0] >> pt[1] >> pt[2];

      ptvec.push_back(pt); 
      counter++;
    }
  }
  return counter;
}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::Report()
{
  std::vector<std::string> fnvec;

  boost::regex filter("(.+)(.txt)", boost::regex::icase);
  for ( boost::filesystem::directory_iterator endItr, it(m_OutputDirectory); it != endItr; ++it )
  {
    if ( boost::regex_match(it->path().string().c_str(), filter) )
    {
      fnvec.push_back(it->path().string());
    }
  }

  std::string fnName = m_OutputDirectory + "\\report.txt";
  std::ofstream ofs;
  ofs.precision(3);
  ofs.open(fnName.c_str(), std::ios::out);

  if (!ofs.is_open())
  {
    std::cout << "Error opening file : " << fnName;
  }

  ofs << "mean dev max min" << '\n';

  std::vector<float*> intrinsicvec;
  std::vector<float*> distortionvec;

  for ( int fn=0; fn<fnvec.size(); fn++ )
  {
    this->ReadIntrinsicAndDistortionFromFile(fnvec[fn].c_str(), intrinsicvec, distortionvec);

    ofs << '\n' << fnvec[fn].c_str() << '\n';

    std::vector<cv::Scalar> reporter;
    for ( int j=0; j<9; j++)
    {
      std::vector<float> tmp;
      for ( int i=0; i<intrinsicvec.size(); i++ )
      {
        tmp.push_back(intrinsicvec[i][j]);
      }
      cv::Scalar mean, dev;
      cv::meanStdDev(tmp, mean, dev);

      cv::Scalar result;
      result(0) = mean(0);
      result(1) = dev(0);
      result(2) = *std::max_element(tmp.begin(), tmp.end());
      result(3) = *std::min_element(tmp.begin(), tmp.end());

      reporter.push_back(result);
    }

    for ( int j=0; j<4; j++)
    {
      std::vector<float> tmp;
      for ( int i=0; i<distortionvec.size(); i++ )
      {
        tmp.push_back(distortionvec[i][j]);
      }
      cv::Scalar mean, dev;
      cv::meanStdDev(tmp, mean, dev);

      cv::Scalar result;
      result(0) = mean(0);
      result(1) = dev(0);
      result(2) = *std::max_element(tmp.begin(), tmp.end());
      result(3) = *std::min_element(tmp.begin(), tmp.end());

      reporter.push_back(result);
    }
    
    for ( int k=0; k<reporter.size(); k++ )
    {
      ofs << reporter[k](0) << " " << reporter[k](1) << " " << reporter[k](2) << " " << reporter[k](3) << '\n';
    }

    intrinsicvec.clear();
    distortionvec.clear();
  }

  ofs.close();
}

//---------------------------------------------------------------------------
void EvaluateIntrinsicParametersOnNumberOfFrames::ReadIntrinsicAndDistortionFromFile(const char *fn, 
                                                                                    std::vector<float*> &intrinsicvec,
                                                                                    std::vector<float*> &distortionvec)
{
  std::fstream fp( fn, std::ios::in ) ;
  
	if( !fp.is_open() )
	{
		std::cout << "In ReadIntrinsicAndDistortionFromFile, file can't be opened";
		return;
	}

  bool is_mat2 = false;
  int counter = 0;
  float intrinsic[9];
  float distortion[4];
  do
	{
    std::string line;
		getline(fp, line);

    if ( line != "" )
    {
      
      char* pch;
      pch = strtok( (char*)(line.c_str()), " [],;\t\n"); 
      
      while ( (pch != NULL) && (!is_mat2) )
      {
        intrinsic[counter] = atof(pch);
        counter++;

        if ( counter == 9 )
        {
          float* tmp = new float[9];
          for ( int i=0; i<9; i++ )
          {
            tmp[i] = intrinsic[i];
          }

          intrinsicvec.push_back(tmp); 
          is_mat2 = true;
          counter = 0;
        }
        pch  = strtok( NULL, " [],;\t\n");
      }

      while ( (pch != NULL) && is_mat2 )
      {
        distortion[counter] = atof(pch);
        counter++;

        if ( counter == 4 )
        {
          float* tmp = new float[4];
          for ( int i=0; i<4; i++ )
          {
            tmp[i] = distortion[i];
          }

          distortionvec.push_back(tmp); 
          is_mat2 = false;
          counter = 0;
        }
        pch  = strtok( NULL, " [],;\t\n");
      }
    }
	}
  while (fp.peek()!=EOF);

	fp.close();
}
 
} // end namespace
