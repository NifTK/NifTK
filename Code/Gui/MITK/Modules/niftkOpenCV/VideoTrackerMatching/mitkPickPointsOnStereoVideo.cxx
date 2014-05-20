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
, m_StartFrame(0)
, m_EndFrame(0)
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

  cvNamedWindow ("Left Channel", CV_WINDOW_AUTOSIZE);
  cvNamedWindow ("Right Channel", CV_WINDOW_AUTOSIZE);
  int framenumber = 0 ;
  int key = 0;
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
      MITK_INFO << framenumber << " " << timingError;
      
      key = cvWaitKey (20);

      std::vector <cv::Point2d> leftPickedPoints;
      unsigned int leftLastPointCount = leftPickedPoints.size() + 1;
      std::vector <cv::Point2d> rightPickedPoints;
      unsigned int rightLastPointCount = rightPickedPoints.size() + 1;
      //if ( framenumber %2 == 0 ) 
      while ( key != 'n' )
      {
        //might need an explicit copy here
        cv::Mat leftAnnotatedVideoImage = leftVideoImage.clone();
        cvSetMouseCallback("Left Channel",CallBackFunc, &leftPickedPoints);
        key = cvWaitKey(20);
        if ( leftPickedPoints.size() != leftLastPointCount )
        {
          for ( int i = 0 ; i < leftPickedPoints.size() ; i ++ ) 
          {
            cv::circle(leftAnnotatedVideoImage, leftPickedPoints[i],5,cv::Scalar(255,255,255),1,1);
          }
            
          IplImage image(leftAnnotatedVideoImage);
          cvShowImage("Left Channel" , &image);
          leftLastPointCount = leftPickedPoints.size();
        }
        cv::Mat rightAnnotatedVideoImage = rightVideoImage.clone();
        cvSetMouseCallback("Right Channel",CallBackFunc, &rightPickedPoints);
        if ( rightPickedPoints.size() != rightLastPointCount )
        {
          for ( int i = 0 ; i < rightPickedPoints.size() ; i ++ ) 
          {
            cv::circle(rightAnnotatedVideoImage, rightPickedPoints[i],5,cv::Scalar(255,255,255),1,1);
          }
            
          IplImage rimage(rightAnnotatedVideoImage);
          cvShowImage("Right Channel" , &rimage);
          rightLastPointCount = rightPickedPoints.size();
        }
      }
      unsigned long long timeStamp;
      trackerMatcher->GetVideoFrame(framenumber, &timeStamp);
      std::string outName = boost::lexical_cast<std::string>(timeStamp) + "_leftPoints.txt";
      std::ofstream pointOut (outName.c_str());
      pointOut << "# " << framenumber << std::endl;
      for ( int i = 0 ; i < leftPickedPoints.size(); i ++ ) 
      {
        pointOut << leftPickedPoints[i] << std::endl;
      }
      pointOut.close();
      trackerMatcher->GetVideoFrame(framenumber+1, &timeStamp);
      outName = boost::lexical_cast<std::string>(timeStamp) + "_rightPoints.txt";
      std::ofstream rightPointOut (outName.c_str());
      rightPointOut << "# " << framenumber+1 << std::endl;
      for ( int i = 0 ; i < rightPickedPoints.size(); i ++ ) 
      {
        rightPointOut << rightPickedPoints[i] << std::endl;
      }
      rightPointOut.close();

      exit(1);
    }
    framenumber += 2;
  }
  m_ProjectOK = true;
}
//-----------------------------------------------------------------------------
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
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


//-----------------------------------------------------------------------------
std::vector < std::vector <cv::Point3d> > PickPointsOnStereoVideo::GetPointsInLeftLensCS()
{
  std::vector < std::vector < cv::Point3d > > returnPoints;
  for ( unsigned int i = 0 ; i < m_PointsInLeftLensCS.size() ; i ++ ) 
  {
    std::vector < cv::Point3d > thesePoints;
    for ( unsigned int j = 0 ; j < m_PointsInLeftLensCS[i].second.size() ; j ++ ) 
    {
      thesePoints.push_back ( m_PointsInLeftLensCS[i].second[j].first );
    }
    returnPoints.push_back(thesePoints);
  }
  return returnPoints;
}

} // end namespace
