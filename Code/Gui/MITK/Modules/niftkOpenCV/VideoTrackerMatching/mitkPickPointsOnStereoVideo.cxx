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
#include <mitkOpenCVFileIOUtils.h>
#include <cv.h>
//#include <opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include <niftkFileHelper.h>

#include <boost/filesystem.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
PickedObject::PickedObject()
: id (-1)
, isLine (false)
, ordered (false)
{}

//-----------------------------------------------------------------------------
PickedObject::~PickedObject()
{}

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
, m_PickingLine(false)
, m_AskOverWrite(false)
, m_HaltOnVideoReadFail(true)
, m_WriteAnnotatedImages(false)
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
    try
    {
      m_Capture = mitk::InitialiseVideoCapture(m_VideoIn, (! m_HaltOnVideoReadFail )); 
    }
    catch (std::exception& e)
    {
      MITK_ERROR << "Caught exception " << e.what();
      exit(1);
    }
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
      cv::Mat videoImage;
      m_Capture->read(videoImage);
      MITK_INFO << "Skipping frame " << framenumber;
      framenumber ++;
    }
    else
    {
      long long timingError;
      cv::Mat WorldToLeftCamera = trackerMatcher->GetCameraTrackingMatrix(framenumber, &timingError, m_TrackerIndex, NULL, m_ReferenceIndex).inv();

      cv::Mat tempMat;
      cv::Mat leftVideoImage;
      m_Capture->read(tempMat);
      leftVideoImage = tempMat.clone();
      m_Capture->read(tempMat);
      cv::Mat rightVideoImage = tempMat.clone();

      if ( std::abs(timingError) <  m_AllowableTimingError )
      {
        key = cvWaitKey (20);

        std::vector <PickedObject*> leftPickedPoints;
        unsigned int leftLastPointCount = leftPickedPoints.size() + 1;
        std::vector <PickedObject*> rightPickedPoints;
        unsigned int rightLastPointCount = rightPickedPoints.size() + 1;
        //init vectors with empty last element, containing the current point picking state
        leftPickedPoints.push_back(PickedObject::New());
        rightPickedPoints.push_back(PickedObject::New());
        leftPickedPoints.back()->ordered=m_OrderedPoints;
        leftPickedPoints.back()->isLine=m_PickingLine;
        rightPickedPoints.back()->ordered=m_OrderedPoints;
        rightPickedPoints.back()->isLine=m_PickingLine;
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
          
          std::string leftOutName = boost::lexical_cast<std::string>(timeStamp) + "_leftPoints";
          trackerMatcher->GetVideoFrame(framenumber+1, &timeStamp);
          std::string rightOutName = boost::lexical_cast<std::string>(timeStamp) + "_rightPoints";
          bool overWriteLeft = true;
          bool overWriteRight = true;
          key = 0;
          if ( boost::filesystem::exists (leftOutName + ".xml") )
          {
            if ( m_AskOverWrite )
            {
              MITK_INFO << leftOutName + ".xml" << " exists, overwrite (y/n)";
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
              MITK_INFO << leftOutName + ".xml" << " exists, skipping.";
              overWriteLeft = false;
            }
          }

          key = 0;
          if ( boost::filesystem::exists (rightOutName + ".xml") )
          {
            if ( m_AskOverWrite ) 
            {
              MITK_INFO << rightOutName  + ".xml" << " exists, overwrite (y/n)";
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
              MITK_INFO << rightOutName + ".xml" << " exists, skipping.";
              overWriteRight = false;
            }
          }
         
          cv::Mat leftAnnotatedVideoImage;
          cv::Mat rightAnnotatedVideoImage;
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
                leftAnnotatedVideoImage = leftVideoImage.clone();
                cvSetMouseCallback("Left Channel",PointPickingCallBackFunc, &leftPickedPoints);
                if ( leftPickedPoints.size() != leftLastPointCount )
                {
                  for ( int i = 0 ; i < leftPickedPoints.size() ; i ++ ) 
                  {
                    std::string number;
                    if ( leftPickedPoints[i]->id == -1 )
                    {
                      number = "#";
                    }
                    else 
                    {
                      number = boost::lexical_cast<std::string>(leftPickedPoints[i]->id);
                    }
                    for ( unsigned int j = 0 ; j < leftPickedPoints[i]->points.size() ; j ++ )
                    {
                      cv::putText(leftAnnotatedVideoImage,number,leftPickedPoints[i]->points[j],0,1.0,cv::Scalar(255,255,255));
                      cv::circle(leftAnnotatedVideoImage, leftPickedPoints[i]->points[j],5,cv::Scalar(255,255,255),1,1);
                    }
                  }
                
                  IplImage image(leftAnnotatedVideoImage);
                  cvShowImage("Left Channel" , &image);
                  leftLastPointCount = leftPickedPoints.size();
                }
                
              }
              if ( overWriteRight )
              {
                rightAnnotatedVideoImage = rightVideoImage.clone();
                cvSetMouseCallback("Right Channel",PointPickingCallBackFunc, &rightPickedPoints);
                if ( rightPickedPoints.size() != rightLastPointCount )
                {
                  for ( int i = 0 ; i < rightPickedPoints.size() ; i ++ ) 
                  {
                    std::string number;
                    if ( rightPickedPoints[i]->id == -1 )
                    {
                      number = "#";
                    }
                    else
                    {
                      number = boost::lexical_cast<std::string>(i);
                    }
                    for ( unsigned int j = 0 ; j < rightPickedPoints[i]->points.size() ; j ++ )
                    {
                      cv::putText(rightAnnotatedVideoImage,number,rightPickedPoints[i]->points[j],0,1.0,cv::Scalar(255,255,255));
                      cv::circle(rightAnnotatedVideoImage, rightPickedPoints[i]->points[j],5,cv::Scalar(255,255,255),1,1);
                    }
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
            std::ofstream leftPointOut ((leftOutName+ ".xml").c_str());
            leftPointOut << "<frame>" <<  framenumber << "</frame>" << std::endl;
            leftPointOut << "<channel>left</channel>" << std::endl;

            for ( int i = 0 ; i < leftPickedPoints.size(); i ++ ) 
            {
              if ( leftPickedPoints[i]->isLine )
              {
                leftPointOut << "<line>" << std::endl;
                leftPointOut << "<id>" << leftPickedPoints[i]->id << "</id>" << std::endl;
                leftPointOut << "<coordinates>" <<std::endl;
                for ( unsigned int j = 0 ; j < leftPickedPoints[i]->points.size() ; j ++ )
                {
                  leftPointOut << leftPickedPoints[i]->points[j];
                }
                leftPointOut << "</coordinates>" <<std::endl;
                leftPointOut << "</line>" << std::endl;
              }
              else
              {
                leftPointOut << "<point>" << std::endl;
                leftPointOut << "<id>" << leftPickedPoints[i]->id << "</id>" << std::endl;
                leftPointOut << "<coordinates>" <<std::endl;
                for ( unsigned int j = 0 ; j < leftPickedPoints[i]->points.size() ; j ++ )
                {
                  leftPointOut << leftPickedPoints[i]->points[j];
                }
                leftPointOut << "</coordinates>" <<std::endl;
                leftPointOut << "</point>" << std::endl;
              }
            }
            leftPointOut.close();
            if ( m_WriteAnnotatedImages )
            {
              for ( int i = 0 ; i < leftPickedPoints.size(); i ++ )
              {
                std::string number = "#";
                if ( leftPickedPoints[i]->id != -1 )
                {
                  number = boost::lexical_cast<std::string>(i);
                }
                for ( unsigned int j = 0 ; j < leftPickedPoints[i]->points.size() ; j ++ )
                {
                  cv::putText(leftAnnotatedVideoImage,number,leftPickedPoints[i]->points[j],0,1.0,cv::Scalar(255,255,255));
                  cv::circle(leftAnnotatedVideoImage, leftPickedPoints[i]->points[j],5,cv::Scalar(255,255,255),1,1);
                }
              }
              cv::imwrite(leftOutName + ".png" ,leftAnnotatedVideoImage);
            }
          }
          if ( rightPickedPoints.size() != 0 ) 
          {
            std::ofstream rightPointOut ((rightOutName + ".xml").c_str());
            rightPointOut << "<frame>" <<  framenumber << "</frame>" << std::endl;
            rightPointOut << "<channel>right</channel>" << std::endl;

            for ( int i = 0 ; i < rightPickedPoints.size(); i ++ ) 
            {
              if ( rightPickedPoints[i]->isLine )
              {
                rightPointOut << "<line>" << std::endl;
                rightPointOut << "<id>" << rightPickedPoints[i]->id << "</id>" << std::endl;
                rightPointOut << "<coordinates>" <<std::endl;
                for ( unsigned int j = 0 ; j < rightPickedPoints[i]->points.size() ; j ++ )
                {
                  rightPointOut << rightPickedPoints[i]->points[j];
                }
                rightPointOut << "</coordinates>" <<std::endl;
                rightPointOut << "</line>" << std::endl;
              }
              else
              {
                rightPointOut << "<point>" << std::endl;
                rightPointOut << "<id>" << rightPickedPoints[i]->id << "</id>" << std::endl;
                rightPointOut << "<coordinates>" <<std::endl;
                for ( unsigned int j = 0 ; j < rightPickedPoints[i]->points.size() ; j ++ )
                {
                  rightPointOut << rightPickedPoints[i]->points[j];
                }
                rightPointOut << "</coordinates>" <<std::endl;
                rightPointOut << "</point>" << std::endl;
              }
            }
            rightPointOut.close();
            if ( m_WriteAnnotatedImages )
            {
              for ( int i = 0 ; i < rightPickedPoints.size(); i ++ )
              {
                std::string number = "#";
                if ( rightPickedPoints[i]->id != -1 )
                {
                  number = boost::lexical_cast<std::string>(i);
                }
                for ( unsigned int j = 0 ; j < rightPickedPoints[i]->points.size() ; j ++ )
                {
                  cv::putText(rightAnnotatedVideoImage,number,rightPickedPoints[i]->points[j],0,1.0,cv::Scalar(255,255,255));
                  cv::circle(rightAnnotatedVideoImage, rightPickedPoints[i]->points[j],5,cv::Scalar(255,255,255),1,1);
                }
              }
              cv::imwrite(rightOutName + ".png" ,rightAnnotatedVideoImage);
            } 
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
  std::vector<PickedObject*>* out = static_cast<std::vector<PickedObject*>*>(userdata);
  if  ( event == cv::EVENT_LBUTTONDOWN )
  {
    if ( out->back()->isLine ) 
    {
      //do something
    }
    else
    {
      if ( out->back()->ordered )
      {
        int lastPoint = 0 ; 
        for ( unsigned int i = 0 ; i < out->size() - 1 ; i ++ ) 
        {
          if ( ! (*out)[i]->isLine ) 
          {
            if ( (*out)[i]->id > lastPoint ) 
            {
              lastPoint = (*out)[i]->id;
            }
          }
        }
        out->back()->points.push_back(cv::Point2d(x,y));
        out->back()->id = lastPoint;
        PickedObject::Pointer pickedObject = PickedObject::New();
        pickedObject->isLine = out->back()->isLine;
        pickedObject->ordered = out->back()->ordered;
        out->push_back(pickedObject);
        MITK_INFO << "Picked ordered point " << lastPoint+1 << " , " <<  cv::Point2d(x,y);
      }
    }
  }
  else if  ( event == cv::EVENT_RBUTTONDOWN )
  {
    if ( out->size() > 0 ) 
    { 
      out->pop_back();
      MITK_INFO << "Removed last picked object" << out->size();
    }
  }
  else if  ( event == cv::EVENT_MBUTTONDOWN )
  {
    MITK_INFO << "Skipping Point " << out->size();
  //  out->push_back(cv::Point2d(-1,-1));
  }

}


} // end namespace
