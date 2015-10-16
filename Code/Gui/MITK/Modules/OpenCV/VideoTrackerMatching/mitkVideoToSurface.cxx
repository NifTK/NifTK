/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoToSurface.h"
#include <niftkSequentialCpuQds.h>
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVPointTypes.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkPointSetWriter.h>
#include <cv.h>
#include <highgui.h>
#include <niftkFileHelper.h>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/lexical_cast.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
VideoToSurface::VideoToSurface()
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_Directory("")
, m_TrackerIndex(0)
, m_ReferenceIndex(-1)
, m_InitOK(false)
, m_HaltOnVideoReadFail(true)
, m_VisualiseTrackingStatus(false)
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
, m_LeftWriter(NULL)
, m_RightWriter(NULL)
, m_AllowableTimingError (20e6) // 20 milliseconds 
, m_StartFrame(0)
, m_EndFrame(0)
{
}


//-----------------------------------------------------------------------------
VideoToSurface::~VideoToSurface()
{

}

//-----------------------------------------------------------------------------
void VideoToSurface::SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer trackerMatcher)
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
void VideoToSurface::Initialise(std::string directory, 
    std::string calibrationParameterDirectory)
{
  m_InitOK = false;

  Initialise ( directory );
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

  return;
}

//-----------------------------------------------------------------------------
void VideoToSurface::Initialise(std::string directory)
{
  m_InitOK = false;
  m_Directory = directory;
  
  m_OutDirectory = m_Directory + niftk::GetFileSeparator() +  "Surface_Recon_Results";
 
  m_InitOK = true;
  return;
}

//-----------------------------------------------------------------------------
void VideoToSurface::FindVideoData(mitk::VideoTrackerMatching::Pointer trackerMatcher) 
{
  if ( m_Capture == NULL ) 
  {
    m_VideoIn = niftk::FindVideoFile ( m_Directory , niftk::Basename (niftk::Basename ( trackerMatcher->GetFrameMap() )));
    if ( m_VideoIn == "" )
    {
      m_InitOK = false;
      return;
    }
    try
    {
      m_Capture = mitk::InitialiseVideoCapture(m_VideoIn, ( ! m_HaltOnVideoReadFail ));
    }
    catch (std::exception& e)
    {
      MITK_ERROR << "Caught exception " << e.what();
      exit(1);
    }
  }
  
  if ( m_SaveVideo ) 
  {
    try
    {
      niftk::CreateDirAndParents ( m_OutDirectory );
    }
    catch (std::exception& e)
    {
      MITK_ERROR << "Caught exception " << e.what();
    }
    
    cv::Size S = cv::Size((int) m_VideoWidth/2.0, (int) m_VideoHeight );
    double fps = static_cast<double>(m_Capture->get(CV_CAP_PROP_FPS));
    double halfFPS = fps/2.0;
    m_LeftWriter = cvCreateVideoWriter(std::string( m_OutDirectory + niftk::Basename(m_VideoIn) +  "_leftchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),halfFPS,S, true);
  }

  return;
}

//-----------------------------------------------------------------------------
void VideoToSurface::SetSaveVideo ( bool savevideo )
{
  if ( m_InitOK ) 
  {
    MITK_WARN << "Changing save video  state after initialisation, will need to re-initialise";
  }
  m_SaveVideo = savevideo;

  m_InitOK = false;
  return;
}

//-----------------------------------------------------------------------------
void VideoToSurface::Reconstruct(mitk::VideoTrackerMatching::Pointer trackerMatcher, 
    std::vector<double>* perturbation)
{
  if ( ! m_InitOK )
  {
    MITK_WARN << "Called project before initialise.";
    return;
  }
 
  this->FindVideoData(trackerMatcher);

  int framenumber = 0 ;
  int key = 0;

  niftk::SequentialCpuQds  featureMatcher(m_PatchWidth, m_PatchHeight);
  while ( framenumber < trackerMatcher->GetNumberOfFrames() ) 
  {
    cv::Mat leftImage;
    cv::Mat rightImage;
    m_Capture->read(leftImage);
    m_Capture->read(rightImage);
    if ( ( m_StartFrame < m_EndFrame ) && ( framenumber < m_StartFrame || framenumber > m_EndFrame ) )
    {
      MITK_INFO << "Skipping frames " << framenumber << " and " << framenumber + 1;
      framenumber ++;
      framenumber ++;
    }
    else
    {
      CorrectDistortionInSingleImage ( m_LeftIntrinsicMatrix, m_LeftDistortionVector, 
          leftImage );
      CorrectDistortionInSingleImage ( m_RightIntrinsicMatrix, m_RightDistortionVector, 
          rightImage );

      long long timingError;
      cv::Mat WorldToLeftCamera = trackerMatcher->GetCameraTrackingMatrix(framenumber, &timingError, m_TrackerIndex, perturbation, m_ReferenceIndex).inv();
      
      unsigned long long matrixTimeStamp;
      unsigned long long absTimingError = static_cast<unsigned long long> ( abs(timingError));
      if ( timingError < 0 ) 
      {
        matrixTimeStamp = trackerMatcher->GetVideoFrameTimeStamp(framenumber) + absTimingError;
      }
      else
      {
        matrixTimeStamp = trackerMatcher->GetVideoFrameTimeStamp(framenumber) - absTimingError;
      }

      m_WorldToLeftCameraMatrices.push_back(WorldToLeftCamera);

      cv::Mat leftPatch = this->GetPatch (leftImage);
      cv::Mat rightPatch = this->GetPatch (rightImage);

      featureMatcher.Process ( leftPatch, rightPatch ) ;
    
      cv::Mat disparityImage = featureMatcher.CreateDisparityImage();
      
      std::vector < std::pair < cv::Point2d , cv::Point2d > > matchedPairs;

      for ( unsigned int row = 0 ; row < m_PatchHeight ; ++row )
      {
        for ( unsigned int column = 0 ; column < m_PatchWidth ; ++column )
        {
          cv::Point2d match = featureMatcher.GetMatch ( column, row );

          if ( match.x != 0 )
          {
            matchedPairs.push_back ( std::pair <cv::Point2d, cv::Point2d> 
                ( cv::Point2d ( column + m_PatchOriginX, row + m_PatchOriginY ) ,
                  cv::Point2d ( match.x + m_PatchOriginX, match.y + m_PatchOriginY )) ); 
          }
        }
      }
    
      std::vector < std::pair < cv::Point3d, double > > triangulatedPoints = 
        mitk::TriangulatePointPairsUsingGeometry ( matchedPairs ,
            m_LeftIntrinsicMatrix, m_RightIntrinsicMatrix, 
            m_RightToLeftRotationMatrix, m_RightToLeftTranslationVector,
            m_TriagulationTolerance );
   
      std::vector <cv::Point3d> points;
      double meanError = 0;
      std::vector <unsigned int> histogram;
      for ( unsigned int i = 0 ; i < m_HistogramMaximumDepth + 1 ; i ++ )
      {
        histogram.push_back(0);
      }
      
      for ( std::vector < std::pair <cv::Point3d, double> >::iterator it = triangulatedPoints.begin() ; it <  triangulatedPoints.end() ; ++it )
      {
        unsigned int bin = static_cast<unsigned int> ( floor ( it->first.z + 0.5 ) );
        if ( bin > m_HistogramMaximumDepth ) 
        {
          bin = m_HistogramMaximumDepth;
        }
        histogram[bin]++;

        points.push_back ( it->first );
        errors.push_back ( it->second );
        
        meanError += it->second;
      }
      meanError /= static_cast<double>( triangulatedPoints.size();

      cv::Point3d stddev;

      cv::Point3d centroid = mitk::GetCentroid ( points, false, &stddev );
   
      this->AnnoateImage ( leftImage, dispariityImage, timingError, centroid.z, stddev.z,
        triangulatedPoints.size(), histogram, meanError );

    }
  }
}


//-----------------------------------------------------------------------------
/*cv::Mat VideoToSurface::AnnotateImage(mitk::VideoTrackerMatching::Pointer trackerMatcher, 
      {
        cv::Mat videoImage;
        m_Capture->read(videoImage);
        if ( drawProjection )
        {
          m_ProjectedPointLists.back()->AnnotateImage(videoImage);
        }  
        if ( m_DrawAxes )
        {
          if ( framenumber % 2 == 0 )
          {
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Left,m_ScreenAxesPoints.m_Points[1].m_Left,cvScalar(255,0,0));
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Left,m_ScreenAxesPoints.m_Points[2].m_Left,cvScalar(0,255,0));
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Left,m_ScreenAxesPoints.m_Points[3].m_Left,cvScalar(0,0,255));         
          }
          else
          {
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Right,m_ScreenAxesPoints.m_Points[1].m_Right,cvScalar(255,0,0));
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Right,m_ScreenAxesPoints.m_Points[2].m_Right,cvScalar(0,255,0));
            cv::line(videoImage,m_ScreenAxesPoints.m_Points[0].m_Right,m_ScreenAxesPoints.m_Points[3].m_Right,cvScalar(0,0,255));         
          }
        }
        if ( m_VisualiseTrackingStatus )
        {
          unsigned int howMany = trackerMatcher->GetTrackingMatricesSize();

          for ( unsigned int i = 0 ; i < howMany ; i ++ ) 
          {

            long long timingError;
            trackerMatcher->GetCameraTrackingMatrix(framenumber , &timingError , i);
            cv::Point2d textLocation = cv::Point2d ( m_VideoWidth - ( m_VideoWidth * 0.03 ) , (i+1) *  m_VideoHeight * 0.07  );
            cv::Point2d location = cv::Point2d ( m_VideoWidth - ( m_VideoWidth * 0.035 ) , (i) *  m_VideoHeight * 0.07 + m_VideoHeight * 0.02  );
            cv::Point2d location1 = cv::Point2d ( m_VideoWidth - ( m_VideoWidth * 0.035 ) + ( m_VideoWidth * 0.025 ) , 
              (i) *  m_VideoHeight * 0.07 + (m_VideoHeight * 0.06) + m_VideoHeight * 0.02);
            if ( timingError < m_AllowableTimingError )
            {
              cv::rectangle ( videoImage, location, location1  , cvScalar (0,255,0), CV_FILLED);
              cv::putText(videoImage , "T" + boost::lexical_cast<std::string>(i), textLocation ,0,1.0, cvScalar ( 255,255,255), 4.0);
            }
            else
            {
              cv::rectangle ( videoImage, location, location1  , cvScalar (0,0,255), CV_FILLED);
              cv::putText(videoImage , "T" + boost::lexical_cast<std::string>(i), textLocation ,0,1.0, cvScalar ( 255,255,255), 4.0);
            }
          }
        }
        if ( m_AnnotateWithGoldStandards )
        {
          std::vector < mitk::PickedObject > goldStandardObjects;
          for ( std::vector<mitk::PickedObject>::iterator it = m_GoldStandardPoints.begin()  ; it < m_GoldStandardPoints.end() ; ++it ) 
          {
            if ( it->m_FrameNumber == framenumber ) 
            {
              if ( framenumber%2 == 0 )
              {
                if ( it->m_Channel == "left" )
                {
                  goldStandardObjects.push_back(*it);
                  goldStandardObjects.back().m_Scalar = cv::Scalar ( 0,255,255);
                }
              }
              if ( framenumber%2 !=0 )
              {
                if ( it->m_Channel == "right" )
                {
                  goldStandardObjects.push_back(*it);
                  goldStandardObjects.back().m_Scalar = cv::Scalar ( 0,255,255);
                }
              }
            }
          }
          if ( goldStandardObjects.size() != 0 )
          {
            mitk::PickedPointList::Pointer goldStandardPointList = mitk::PickedPointList::New();
            goldStandardPointList->SetPickedObjects(goldStandardObjects);
            goldStandardPointList->AnnotateImage(videoImage);
            if ( m_WriteAnnotatedGoldStandards ) 
            {
              std::string outname = boost::lexical_cast<std::string>(framenumber) + ".png";
              cv::imwrite(outname,videoImage);
            }

          }
        }

        if ( m_SaveVideo )
        {
          if ( m_LeftWriter != NULL ) 
          {
            if ( framenumber%2 == 0 ) 
            {
              IplImage image(videoImage);
              cvResize (&image, smallcorrectedimage,CV_INTER_LINEAR);
              cvWriteFrame(m_LeftWriter,smallcorrectedimage);
            }
          }
          if ( m_RightWriter != NULL ) 
          {
            if ( framenumber%2 != 0 ) 
            {
              IplImage image(videoImage);
              cvResize (&image, smallcorrectedimage,CV_INTER_LINEAR);
              cvWriteFrame(m_RightWriter,smallcorrectedimage);
            }
          }
        }
        if ( m_Visualise ) 
        {
          IplImage image(videoImage);
          cvResize (&image, smallimage,CV_INTER_LINEAR);
          if ( framenumber %2 == 0 ) 
          {
            cvShowImage("Left Channel" , smallimage);
          }
          else
          {
            cvShowImage("Right Channel" , smallimage);
          }
          key = cvWaitKey (20);
          if ( key == 's' )
          {
            m_Visualise = false;
          }
          if ( key == 't' )
          {
            drawProjection = ! drawProjection;
          }
        }
      }
      framenumber ++;
    }
  }
  if ( m_LeftWriter != NULL )
  {
    cvReleaseVideoWriter(&m_LeftWriter);
  }
  if ( m_RightWriter != NULL )
  {
    cvReleaseVideoWriter(&m_RightWriter);
  }
  m_ProjectOK = true;

}*/

} // end namespace
