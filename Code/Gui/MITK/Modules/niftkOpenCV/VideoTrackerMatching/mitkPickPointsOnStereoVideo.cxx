/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPickPointsOnStereoVideo.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <cv.h>
//#include <opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include <niftkFileHelper.h>

#include <boost/filesystem.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
PickPointsOnStereoVideo::PickPointsOnStereoVideo()
: 
m_VideoIn("")
, m_Directory("")
, m_TrackerIndex(0)
, m_ReferenceIndex(-1)
, m_InitOK(false)
, m_ProjectOK(false)
, m_LeftIntrinsicMatrix (new cv::Mat(3,3,CV_64FC1))
, m_LeftDistortionVector (new cv::Mat(1,4,CV_64FC1))
, m_RightIntrinsicMatrix (new cv::Mat(3,3,CV_64FC1))
, m_RightDistortionVector (new cv::Mat(1,4,CV_64FC1))
, m_RightToLeftRotationMatrix (new cv::Mat(3,3,CV_64FC1))
, m_RightToLeftTranslationVector (new cv::Mat(3,1,CV_64FC1))
, m_LeftCameraToTracker (new cv::Mat(4,4,CV_64FC1))
, m_VideoWidth(1920)
, m_VideoHeight(540)
, m_Capture(NULL)
, m_AllowableTimingError (20e6) // 20 milliseconds 
, m_OrderedPoints(false)
, m_AskOverWrite(false)
, m_StartFrame(0)
, m_EndFrame(0)
, m_Frequency(50)
{
}


//-----------------------------------------------------------------------------
PickPointsOnStereoVideo::~PickPointsOnStereoVideo()
{

}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer trackerMatcher)
{
  if ( ! m_InitOK ) 
  {
    MITK_ERROR << "Can't set trackerMatcher handeye before projector initialiastion";
    return;
  }
  trackerMatcher->SetCameraToTracker(*m_LeftCameraToTracker, m_TrackerIndex);
  return;
}
//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::Initialise(std::string directory, 
    std::string calibrationParameterDirectory)
{
  m_InitOK = false;
  m_Directory = directory;

  try
  {
    mitk::LoadStereoCameraParametersFromDirectory
      ( calibrationParameterDirectory,
      m_LeftIntrinsicMatrix,m_LeftDistortionVector,m_RightIntrinsicMatrix,
      m_RightDistortionVector,m_RightToLeftRotationMatrix,
      m_RightToLeftTranslationVector,m_LeftCameraToTracker);
  }
  catch ( int e )
  {
    MITK_ERROR << "Failed to load camera parameters";
    m_InitOK = false;
    return;
  }
  
  if ( m_Capture == NULL ) 
  {
    std::vector <std::string> videoFiles = niftk::FindVideoData(m_Directory);
    if ( videoFiles.size() == 0 ) 
    {
      MITK_ERROR << "Failed to find any video files";
      m_InitOK = false;
      return;
    }
    if ( videoFiles.size() > 1 ) 
    {
      MITK_WARN << "Found multiple video files, will only use " << videoFiles[0];
    }
    m_VideoIn = videoFiles[0];
   
    m_Capture = cvCreateFileCapture(m_VideoIn.c_str()); 
  }
  
  if ( ! m_Capture )
  {
    MITK_ERROR << "Failed to open " << m_VideoIn;
    m_InitOK=false;
    return;
  }

  m_InitOK = true;
  return;

}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::Project(mitk::VideoTrackerMatching::Pointer trackerMatcher)
{
  if ( ! m_InitOK )
  {
    MITK_WARN << "Called project before initialise.";
    return;
  }
    
  m_ProjectOK = false;

  int framenumber = 0 ;
  int key = 0;
  cvNamedWindow ("Left Channel", CV_WINDOW_AUTOSIZE);
  cvNamedWindow ("Right Channel", CV_WINDOW_AUTOSIZE);
     
  cv::Mat blankMat = cvCreateMat(10,100,CV_32FC3);
  IplImage blankImage(blankMat);
  cvShowImage("Left Channel" , &blankImage);
  cvShowImage("Right Channel" , &blankImage);
  unsigned long long startTime;
  trackerMatcher->GetVideoFrame(0, &startTime);
  while ( framenumber < trackerMatcher->GetNumberOfFrames() && key != 'q')
  {
    if ( ( m_StartFrame < m_EndFrame ) && ( framenumber < m_StartFrame || framenumber > m_EndFrame ) )
    {
      cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
      MITK_INFO << "Skipping frame " << framenumber;
      framenumber ++;
    }
    else
    {
      long long timingError;
      cv::Mat WorldToLeftCamera = trackerMatcher->GetCameraTrackingMatrix(framenumber, &timingError, m_TrackerIndex, NULL, m_ReferenceIndex).inv();

      cv::Mat leftVideoImage = cvQueryFrame ( m_Capture ) ;
      cv::Mat rightVideoImage = cvQueryFrame ( m_Capture ) ;
      if ( std::abs(timingError) <  m_AllowableTimingError )
      {
        key = cvWaitKey (20);

        std::vector <cv::Point2d> leftPickedPoints;
        unsigned int leftLastPointCount = leftPickedPoints.size() + 1;
        std::vector <cv::Point2d> rightPickedPoints;
        unsigned int rightLastPointCount = rightPickedPoints.size() + 1;
        if ( framenumber %m_Frequency == 0 ) 
        {
          unsigned long long timeStamp;
          trackerMatcher->GetVideoFrame(framenumber, &timeStamp);

          if ( m_OrderedPoints )
          {
            MITK_INFO << "Picking ordered points on frame pair " << framenumber << ", " << framenumber+1 << " [ " <<  (timeStamp - startTime)/1e9 << " s ] t to pick unordered, n for next frame, q to quit";
          }
          else 
          {
            MITK_INFO << "Picking un ordered points on frame pair " << framenumber << ", " << framenumber+1 << " [ " << (timeStamp - startTime)/1e9 << " s ] t to pick ordered, n for next frame, q to quit";
          }
          
          std::string leftOutName = boost::lexical_cast<std::string>(timeStamp) + "_leftPoints.txt";
          trackerMatcher->GetVideoFrame(framenumber+1, &timeStamp);
          std::string rightOutName = boost::lexical_cast<std::string>(timeStamp) + "_rightPoints.txt";
          bool overWriteLeft = true;
          bool overWriteRight = true;
          key = 0;
          if ( boost::filesystem::exists (leftOutName) )
          {
            if ( m_AskOverWrite )
            {
              MITK_INFO << leftOutName << " exists, overwrite (y/n)";
              key = 0;
              while ( ! ( key == 'n' || key == 'y' ) )
              {
                key = cvWaitKey(20);
              }
              if ( key == 'n' ) 
              {
                overWriteLeft = false;
              }
              else
              {
                overWriteLeft = true;
              }
            }
            else
            {
              MITK_INFO << leftOutName << " exists, skipping.";
              overWriteLeft = false;
            }
          }

          key = 0;
          if ( boost::filesystem::exists (rightOutName) )
          {
            if ( m_AskOverWrite ) 
            {
              MITK_INFO << rightOutName << " exists, overwrite (y/n)";
              while ( ! ( key == 'n' || key == 'y' ) )
              {
                key = cvWaitKey(20);
              }
              if ( key == 'n' ) 
              {
                overWriteRight = false;
              }
              else
              {
                overWriteRight = true;
              }
            }
            else 
            {
              MITK_INFO << rightOutName << " exists, skipping.";
              overWriteRight = false;
            }
          }
         
          key = 0;
          if ( overWriteLeft  ||  overWriteRight  )
          {
            while ( key != 'n' && key != 'q' )
            {
              key = cvWaitKey(20);
              if ( key == 't' )
              {
                m_OrderedPoints = ! m_OrderedPoints;
                if ( m_OrderedPoints ) 
                {
                  MITK_INFO << "Switched to ordered points mode";
                }
                else
                {
                  MITK_INFO << "Switched to un ordered points mode";
                }
              }
              if ( overWriteLeft )
              {
                cv::Mat leftAnnotatedVideoImage = leftVideoImage.clone();
                cvSetMouseCallback("Left Channel",PointPickingCallBackFunc, &leftPickedPoints);
                if ( leftPickedPoints.size() != leftLastPointCount )
                {
                  for ( int i = 0 ; i < leftPickedPoints.size() ; i ++ ) 
                  {
                    std::string number = boost::lexical_cast<std::string>(i);
                    cv::putText(leftAnnotatedVideoImage,number,leftPickedPoints[i],0,1.0,cv::Scalar(255,255,255));
                    cv::circle(leftAnnotatedVideoImage, leftPickedPoints[i],5,cv::Scalar(255,255,255),1,1);
                  }
                
                  IplImage image(leftAnnotatedVideoImage);
                  cvShowImage("Left Channel" , &image);
                  leftLastPointCount = leftPickedPoints.size();
                }
              }
              if ( overWriteRight )
              {
                cv::Mat rightAnnotatedVideoImage = rightVideoImage.clone();
                cvSetMouseCallback("Right Channel",PointPickingCallBackFunc, &rightPickedPoints);
                if ( rightPickedPoints.size() != rightLastPointCount )
                {
                  for ( int i = 0 ; i < rightPickedPoints.size() ; i ++ ) 
                  {
                    std::string number = boost::lexical_cast<std::string>(i);
                    cv::putText(rightAnnotatedVideoImage,number,rightPickedPoints[i],0,1.0,cv::Scalar(255,255,255));
                    cv::circle(rightAnnotatedVideoImage, rightPickedPoints[i],5,cv::Scalar(255,255,255),1,1);
                  }
                
                  IplImage rimage(rightAnnotatedVideoImage);
                  cvShowImage("Right Channel" , &rimage);
                  rightLastPointCount = rightPickedPoints.size();
                }
              }
            }
          }
          if ( leftPickedPoints.size() != 0 ) 
          {
            std::ofstream leftPointOut (leftOutName.c_str());
            leftPointOut << "# " << framenumber << std::endl;
            if ( m_OrderedPoints ) 
            {
              leftPointOut << "# Ordered" << std::endl;
            }
            else
            {
              leftPointOut << "# UnOrdered" << std::endl;
            }

            for ( int i = 0 ; i < leftPickedPoints.size(); i ++ ) 
            {
              leftPointOut << leftPickedPoints[i] << std::endl;
            }
            leftPointOut.close();
          }
          if ( rightPickedPoints.size() != 0 ) 
          {
            std::ofstream rightPointOut (rightOutName.c_str());
            rightPointOut << "# " << framenumber+1 << std::endl;
            if ( m_OrderedPoints ) 
            {
              rightPointOut << "# Ordered" << std::endl;
            }
            else
            {
              rightPointOut << "# UnOrdered" << std::endl;
            }
            for ( int i = 0 ; i < rightPickedPoints.size(); i ++ ) 
            {
              rightPointOut << rightPickedPoints[i] << std::endl;
            }
            rightPointOut.close();
          }
          cvShowImage("Left Channel" , &blankImage);
          cvShowImage("Right Channel" , &blankImage);
        } 
      }
      else
      {
        MITK_INFO << "Skipping frame " << framenumber << " high timing error " << timingError;
      }
      framenumber += 2;
    }
  }
  m_ProjectOK = true;
}
//-----------------------------------------------------------------------------
void PointPickingCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  std::vector<cv::Point2d>* out = static_cast<std::vector<cv::Point2d>*>(userdata);
  if  ( event == cv::EVENT_LBUTTONDOWN )
  {
    MITK_INFO << "Picked point " << out->size();
    out->push_back (cv::Point2d(x,y));
  }
  else if  ( event == cv::EVENT_RBUTTONDOWN )
  {
    if ( out->size() > 0 ) 
    { 
      out->pop_back();
      MITK_INFO << "Removed point" << out->size();
    }
  }
  else if  ( event == cv::EVENT_MBUTTONDOWN )
  {
    MITK_INFO << "Skipping Point " << out->size();
    out->push_back(cv::Point2d(-1,-1));
  }
}


} // end namespace
