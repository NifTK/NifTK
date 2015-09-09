/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectPointsOnStereoVideo.h"
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
ProjectPointsOnStereoVideo::ProjectPointsOnStereoVideo()
: m_Visualise(false)
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_Directory("")
, m_TriangulatedPointsOutName("")
, m_TrackerIndex(0)
, m_ReferenceIndex(-1)
, m_InitOK(false)
, m_ProjectOK(false)
, m_GoldStandardPointsClassifiedOK(false)
, m_TriangulateOK(false)
, m_DrawAxes(false)
, m_HaltOnVideoReadFail(true)
, m_DontProject(false)
, m_LeftGSFramesAreEven(true)
, m_RightGSFramesAreEven(true)
, m_MaxGoldStandardIndex(-1)
, m_RightGSFrameOffset(0)
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
, m_WorldPoints(NULL)
, m_ClassifierWorldPoints(NULL)
, m_LeftWriter(NULL)
, m_RightWriter(NULL)
, m_AllowablePointMatchingRatio (1.0) 
, m_AllowableTimingError (20e6) // 20 milliseconds 
, m_StartFrame(0)
, m_EndFrame(0)
, m_ProjectorScreenBuffer(0.0)
, m_ClassifierScreenBuffer(100.0)
{
}


//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::~ProjectPointsOnStereoVideo()
{

}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetMatcherCameraToTracker(mitk::VideoTrackerMatching::Pointer trackerMatcher)
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
void ProjectPointsOnStereoVideo::Initialise(std::string directory, 
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
  ProjectAxes(); 

  return;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::Initialise(std::string directory)
{
  m_InitOK = false;
  m_Directory = directory;
  
  m_OutDirectory = m_Directory + niftk::GetFileSeparator() +  "ProjectionResults";
 
  m_InitOK = true;
  return;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::FindVideoData(mitk::VideoTrackerMatching::Pointer trackerMatcher) 
{
  if ( m_Visualise || m_SaveVideo ) 
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
        exit(1);
      }
    }

    if ( m_SaveVideo )
    {
      cv::Size S = cv::Size((int) m_VideoWidth/2.0, (int) m_VideoHeight );
      double fps = static_cast<double>(m_Capture->get(CV_CAP_PROP_FPS));
      double halfFPS = fps/2.0;
      m_LeftWriter =cvCreateVideoWriter(std::string( m_OutDirectory + niftk::Basename(m_VideoIn) +  "_leftchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),halfFPS,S, true);
      m_RightWriter =cvCreateVideoWriter(std::string(m_OutDirectory + niftk::Basename(m_VideoIn) + "_rightchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),halfFPS,S, true);
    }
  }
  return;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetVisualise ( bool visualise )
{
  if ( m_InitOK ) 
  {
    MITK_WARN << "Changing visualisation state after initialisation, will need to re-initialise";
  }
  m_Visualise = visualise;
  m_InitOK = false;
  return;
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetSaveVideo ( bool savevideo )
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
void ProjectPointsOnStereoVideo::Project(mitk::VideoTrackerMatching::Pointer trackerMatcher, 
    std::vector<double>* perturbation)
{
  if ( ! m_InitOK )
  {
    MITK_WARN << "Called project before initialise.";
    return;
  }
 
  this->FindVideoData(trackerMatcher);

  m_ProjectOK = false;
  m_ProjectedPointLists.clear();
  m_PointsInLeftLensCS.clear();
  m_ClassifierProjectedPointLists.clear();
  if ( static_cast<int>(m_WorldPoints->GetListSize()) < m_MaxGoldStandardIndex ) 
  {
    MITK_INFO << "Filling world points with dummy data to enable triangulation";
    cv::Point2i emptyWorldPoint;

    for ( int i = m_WorldPoints->GetListSize() ; i <= m_MaxGoldStandardIndex ; i ++ )
    {
      m_WorldPoints->AddPoint(emptyWorldPoint);
    }
  }
  if ( ! ( m_DontProject ) && ( m_WorldPoints->GetListSize() == 0) )
  {
    MITK_WARN << "Called project with nothing to project";
    return;
  }

  if ( m_Visualise ) 
  {
    cvNamedWindow ("Left Channel", CV_WINDOW_AUTOSIZE);
    cvNamedWindow ("Right Channel", CV_WINDOW_AUTOSIZE);
  }
  int framenumber = 0 ;
  int key = 0;
  bool drawProjection = true;
  IplImage *smallimage = cvCreateImage (cvSize((int)m_VideoWidth/2.0, (int) m_VideoHeight/2.0), 8,3);
  IplImage *smallcorrectedimage = cvCreateImage (cvSize((int)m_VideoWidth/2.0, (int)m_VideoHeight), 8,3);
  while ( framenumber < trackerMatcher->GetNumberOfFrames() && key != 'q')
  {
    if ( ( m_StartFrame < m_EndFrame ) && ( framenumber < m_StartFrame || framenumber > m_EndFrame ) )
    {
      if ( m_Visualise || m_SaveVideo ) 
      {
        cv::Mat videoImage;
        m_Capture->read(videoImage);
        MITK_INFO << "Skipping frame " << framenumber;
      }
      framenumber ++;
    }
    else
    {
      //put the world points into the coordinates of the left hand camera.
      //worldtotracker * trackertocamera
      //in general the tracker matrices are trackertoworld
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

      if ( ! m_DontProject ) 
      {
        m_WorldToLeftCameraMatrices.push_back(WorldToLeftCamera);

        if ( ! m_WorldPoints.IsNull () )
        {
          m_PointsInLeftLensCS.push_back (TransformPickedPointListToLeftLens ( m_WorldPoints, WorldToLeftCamera, matrixTimeStamp, framenumber ));
          m_ProjectedPointLists.push_back( ProjectPickedPointList ( m_PointsInLeftLensCS.back(), m_ProjectorScreenBuffer )) ;
        }
        
        if ( ! m_ClassifierWorldPoints.IsNull() )
        {
          mitk::PickedPointList::Pointer classifierPointsInLeftLensCS = 
            TransformPickedPointListToLeftLens ( m_ClassifierWorldPoints, WorldToLeftCamera, matrixTimeStamp, framenumber );
          m_ClassifierProjectedPointLists.push_back (ProjectPickedPointList ( classifierPointsInLeftLensCS , m_ClassifierScreenBuffer ));
        }
      }

      if ( m_Visualise || m_SaveVideo ) 
      {
        cv::Mat videoImage;
        m_Capture->read(videoImage);
        if ( drawProjection )
        {
          m_ProjectedPointLists.back()->AnnotateImage(videoImage);
          
          if ( m_DrawAxes && drawProjection )
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

}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetLeftGoldStandardPoints (
    std::vector < mitk::GoldStandardPoint > points,
    mitk::VideoTrackerMatching::Pointer matcher )
{
  int maxLeftGSIndex = -1;
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_GoldStandardPoints.push_back(mitk::PickedObject(points[i], matcher->GetVideoFrameTimeStamp ( points[i].m_FrameNumber)));
    m_GoldStandardPoints.back().m_Channel = "left";

    if ( m_GoldStandardPoints.back().m_Id > maxLeftGSIndex ) 
    {
      maxLeftGSIndex =  m_GoldStandardPoints.back().m_Id;
    }
    if ( m_GoldStandardPoints.back().m_FrameNumber % 2 == 0 ) 
    {
      if ( ! m_LeftGSFramesAreEven ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points, left GS should be odd";
        exit(1);
      }
    }
    else
    {
      if ( ( i > 0 ) && ( m_LeftGSFramesAreEven ) ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points, left GS should be even";
        exit(1);
      }
      m_LeftGSFramesAreEven = false;
    }
  }
  if ( m_LeftGSFramesAreEven == m_RightGSFramesAreEven )
  {
    m_RightGSFrameOffset = 0 ;
  }
  else 
  {
    m_RightGSFrameOffset = 1 ;
  }
  if ( maxLeftGSIndex > m_MaxGoldStandardIndex )
  {
    m_MaxGoldStandardIndex = maxLeftGSIndex;
  }
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetGoldStandardObjects( std::vector < mitk::PickedObject > pickedObjects )
{
  int maxLeftGSIndex = -1;
  int maxRightGSIndex = -1;
  if ( m_GoldStandardPoints.size() != 0 )
  {
    MITK_WARN << "Setting gold standard points with non empty vector, check this is what you intended.";
  }
  for ( unsigned int i = 0 ; i < pickedObjects.size() ; i ++ ) 
  {
    if ( pickedObjects[i].m_Channel == "left" )
    {
      if ( pickedObjects[i].m_Id > maxLeftGSIndex )
      {
        maxLeftGSIndex = pickedObjects[i].m_Id;
      }
      if ( pickedObjects[i].m_FrameNumber % 2 == 0 )
      {
        if ( ! m_LeftGSFramesAreEven )
        {
          MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points, left GS should be odd";
          exit(1);
        }
      }
      else
      {
        if ( ( i > 0 ) && ( m_LeftGSFramesAreEven ) ) 
        {
          MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points, left GS should be even";
          exit(1);
        }
        m_LeftGSFramesAreEven = false;
      }
    }
    else
    {
      if ( pickedObjects[i].m_Channel != "right" )
      {
        MITK_ERROR << "Attempted to set gold standard point with unknown channel type " << pickedObjects[i].m_Channel;
        exit(1);
      }
      if ( pickedObjects[i].m_Id > maxRightGSIndex ) 
      {
        maxRightGSIndex =  pickedObjects[i].m_Id;
      }
      if ( pickedObjects[i].m_FrameNumber % 2 == 0 ) 
      {
        if ( ! m_RightGSFramesAreEven ) 
        {
          MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points, right GS should be odd, fn = " <<  pickedObjects[i].m_FrameNumber;
          exit(1);
        }
      }
      else
      {
        if ( ( i > 0 ) && ( m_RightGSFramesAreEven ) ) 
        {
          MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points, right GS should be even";
          exit(1);
        }
        m_RightGSFramesAreEven = false;
      }
    }
    m_GoldStandardPoints.push_back(pickedObjects[i]);
  }

  if ( m_LeftGSFramesAreEven == m_RightGSFramesAreEven )
  {
    m_RightGSFrameOffset = 0 ;
  }
  else 
  {
    m_RightGSFrameOffset = 1 ;
  }
  if ( maxLeftGSIndex > m_MaxGoldStandardIndex )
  {
    m_MaxGoldStandardIndex = maxLeftGSIndex;
  }
  if ( maxRightGSIndex > m_MaxGoldStandardIndex )
  {
    m_MaxGoldStandardIndex = maxRightGSIndex;
  }
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetRightGoldStandardPoints (
    std::vector < mitk::GoldStandardPoint > points ,
    mitk::VideoTrackerMatching::Pointer matcher )
{
  int maxRightGSIndex = -1;
   
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_GoldStandardPoints.push_back(mitk::PickedObject(points[i],  matcher->GetVideoFrameTimeStamp ( points[i].m_FrameNumber)));
    m_GoldStandardPoints.back().m_Channel = "right";
    if ( m_GoldStandardPoints.back().m_Id > maxRightGSIndex ) 
    {
      maxRightGSIndex =  m_GoldStandardPoints[i].m_Id;
    }
    if ( m_GoldStandardPoints.back().m_FrameNumber % 2 == 0 ) 
    {
      if ( ! m_RightGSFramesAreEven ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points, right GS should be odd, fn = " <<  m_GoldStandardPoints[i].m_FrameNumber;
        exit(1);
      }
    }
    else
    {
      if ( ( i > 0 ) && ( m_RightGSFramesAreEven ) ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points, right GS should be even";
        exit(1);
      }
      m_RightGSFramesAreEven = false;
    }
  }
  if ( m_LeftGSFramesAreEven == m_RightGSFramesAreEven )
  {
    m_RightGSFrameOffset = 0 ;
  }
  else 
  {
    m_RightGSFrameOffset = 1 ;
  }
  if ( maxRightGSIndex > m_MaxGoldStandardIndex )
  {
    m_MaxGoldStandardIndex = maxRightGSIndex;
  }
}

//-----------------------------------------------------------------------------
bool ProjectPointsOnStereoVideo::TriangulateGoldStandardObjectList ( )
{
  if ( (! m_ProjectOK ) && (m_MaxGoldStandardIndex == -1 ) ) 
  {
    MITK_ERROR << "Attempted to run CalculateTriangulateErrors, before running project(), no result.";
    return false;
  }
  //everything should already be clasified and sorted so all we need to to do is go through the vector looking for clear();
  //matched pairs

  m_TriangulatedGoldStandardPoints.clear();
  for ( unsigned int i = 0 ; i < m_GoldStandardPoints.size() ; i ++ )
  {
    if ( m_GoldStandardPoints[i].m_Channel == "left" ) 
    {
      for ( unsigned int j = i + 1 ; j < m_GoldStandardPoints.size() ; j ++ ) 
      {
        if ( m_GoldStandardPoints[j].m_FrameNumber == ( m_GoldStandardPoints[i].m_FrameNumber + m_RightGSFrameOffset ) )
        {
          assert ( m_GoldStandardPoints[j].m_Channel == "right" );
          if ( ( m_GoldStandardPoints[i].m_IsLine == m_GoldStandardPoints[j].m_IsLine ) && 
                 (m_GoldStandardPoints[j].m_Id == m_GoldStandardPoints[i].m_Id ) )
          {
            m_TriangulatedGoldStandardPoints.push_back( TriangulatePickedObjects ( m_GoldStandardPoints[i], m_GoldStandardPoints[j] ) );
          }
        }
      }
    }
  }
  return true;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::CalculateTriangulationErrors (std::string outPrefix )
{
  if ( ! m_GoldStandardPointsClassifiedOK )
  {
    ClassifyGoldStandardPoints ();
  }

  if ( ! m_TriangulateOK )
  {
    if ( this->TriangulateGoldStandardObjectList() ) 
    {
       m_TriangulateOK == true;
    }
    else
    {
      MITK_ERROR << "Attempted to run CalculateTriangulateErrors, before running project(), no result.";
      return;
    }
  }
  //everything should already be clasified and sorted so all we need to to do is go through the vector looking for 
  //matched pairs
  m_TriangulationErrors.clear();
  for ( unsigned int i = 0 ; i < m_TriangulatedGoldStandardPoints.size() ; i ++ ) 
  {
    mitk::PickedObject leftLensObject = GetMatchingPickedObject ( m_TriangulatedGoldStandardPoints[i], *m_PointsInLeftLensCS[m_TriangulatedGoldStandardPoints[i].m_FrameNumber] );
    cv::Point3d triangulationError;
    leftLensObject.DistanceTo ( m_TriangulatedGoldStandardPoints[i], triangulationError, m_AllowableTimingError );
    m_TriangulationErrors.push_back ( triangulationError );
  }
  
  std::ofstream tout (std::string (outPrefix + "_triangulation.errors").c_str());
  tout << "#xmm ymm zmm" << std::endl;
  for ( unsigned int i = 0 ; i < m_TriangulationErrors.size() ; i ++ )
  {
    tout << m_TriangulationErrors[i] << std::endl;
  }
  cv::Point3d error3dStdDev;
  cv::Point3d error3dMean;
  double xrms;
  double yrms;
  double zrms;
  double rms;
  error3dMean = mitk::GetCentroid(m_TriangulationErrors, false, &error3dStdDev);
  tout << "#Mean Error      = " << error3dMean << std::endl;
  tout << "#StdDev          = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  zrms = sqrt ( error3dMean.z * error3dMean.z + error3dStdDev.z * error3dStdDev.z );
  rms = sqrt ( xrms*xrms + yrms*yrms + zrms*zrms);
  tout << "#rms             = " << xrms << ", " << yrms << ", " << zrms << ", " << rms << std::endl;
  error3dMean = mitk::GetCentroid(m_TriangulationErrors, true, &error3dStdDev);
  tout << "#Ref. Mean Error = " << error3dMean << std::endl;
  tout << "#Ref. StdDev     = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  zrms = sqrt ( error3dMean.z * error3dMean.z + error3dStdDev.z * error3dStdDev.z );
  rms = sqrt ( xrms*xrms + yrms*yrms + zrms*zrms);
  tout << "#Ref. rms        = " << xrms << ", " << yrms << ", " << zrms << ", " << rms << std::endl;
  tout.close();
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::TriangulateGoldStandardPoints (std::string outPrefix )
{

  if ( ! m_GoldStandardPointsClassifiedOK )
  {
    ClassifyGoldStandardPoints ();
  }

  if ( ! m_TriangulateOK )
  {
    if ( this->TriangulateGoldStandardObjectList() ) 
    {
       m_TriangulateOK == true;
    }
    else
    {
      MITK_ERROR << "Attempted to run CalculateTriangulateErrors, before running project(), no result.";
      return;
    }
  }
  //go through  m_TriangulatedGoldStandardPoints, for each m_Id (points and lines) find the centroid and output 
//if forgot we also use this to triangulate stuff. 
//I guess we want to create a vector of PointLists, into which we can stuff the individual triangulated things.
  mitk::PointSet::Pointer triangulatedPoints = mitk::PointSet::New();
  for ( unsigned int i = 0 ; i < m_TriangulatedGoldStandardPoints.size() ; i ++ ) 
  {
/*    cv::Point3d centroid;
    cv::Point3d stdDev;
    centroid = mitk::GetCentroid (classifiedPoints[i],true, & stdDev);

    mitk::Point3D point;
    point[0] = centroid.x;
    point[1] = centroid.y;
    point[2] = centroid.z;
    triangulatedPoints->InsertPoint(i,point);
    MITK_INFO << "Point " << i << " triangulated mean " << centroid << " SD " << stdDev;*/
  }
  if ( m_TriangulatedPointsOutName != "" )
  {
    mitk::PointSetWriter::Pointer tpWriter = mitk::PointSetWriter::New();
    tpWriter->SetFileName(m_TriangulatedPointsOutName);
    tpWriter->SetInput( triangulatedPoints );
    tpWriter->Update();
  }

}
       
//-----------------------------------------------------------------------------
mitk::PickedObject ProjectPointsOnStereoVideo::TriangulatePickedObjects (mitk::PickedObject po_leftScreen, 
    mitk::PickedObject po_rightScreen )
{
  assert ( po_leftScreen.m_Channel == "left" && po_rightScreen.m_Channel == "right" );
  assert ( po_leftScreen.m_IsLine == po_rightScreen.m_IsLine );

  mitk::PickedObject po_leftLens = po_leftScreen.CopyByHeader();
  po_leftLens.m_Channel = "left_lens";

  if ( po_leftScreen.m_IsLine ) 
  { 
    MITK_WARN << "I don't know how to triangulate a line, giving up";
    return po_leftLens;
  }
  else
  {
    assert ( (po_leftScreen.m_Points[0].z == 0.0) && (po_rightScreen.m_Points[0].z == 0.0) );
    cv::Point2d leftUndistorted;
    cv::Point2d rightUndistorted;
    std::pair < cv::Point2d, cv::Point2d > pair = std::pair < cv::Point2d, cv::Point2d > 
    ( cv::Point2d ( po_leftScreen.m_Points[0].x, po_leftScreen.m_Points[0].y ),
      cv::Point2d ( po_rightScreen.m_Points[0].x, po_rightScreen.m_Points[0].y ));
    bool cropUndistortedPointsToScreen = true;
    double cropValue = std::numeric_limits<double>::quiet_NaN();
    mitk::UndistortPoint(pair.first,
      *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,leftUndistorted,
      cropUndistortedPointsToScreen , 
      0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    mitk::UndistortPoint(pair.second,
      *m_RightIntrinsicMatrix,*m_RightDistortionVector,rightUndistorted,    
      cropUndistortedPointsToScreen , 
      0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
       
    cv::Mat leftCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
    cv::Mat leftCameraRotationVector = cv::Mat (3,1,CV_64FC1);
    cv::Mat rightCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
    cv::Mat rightCameraRotationVector = cv::Mat (3,1,CV_64FC1);

    for ( int j = 0 ; j < 3 ; j ++ )
    {
      leftCameraTranslationVector.at<double>(j,0) = 0.0;
      leftCameraRotationVector.at<double>(j,0) = 0.0;
    }
    rightCameraTranslationVector = *m_RightToLeftTranslationVector * -1;
    cv::Rodrigues ( m_RightToLeftRotationMatrix->inv(), rightCameraRotationVector  );
   
    CvMat* leftScreenPointsMat = cvCreateMat (1,2,CV_64FC1);
    CvMat* rightScreenPointsMat = cvCreateMat (1,2,CV_64FC1);
    CvMat leftCameraIntrinsicMat= *m_LeftIntrinsicMatrix;
    CvMat leftCameraRotationVectorMat= leftCameraRotationVector;
    CvMat leftCameraTranslationVectorMat= leftCameraTranslationVector;
    CvMat rightCameraIntrinsicMat = *m_RightIntrinsicMatrix;
    CvMat rightCameraRotationVectorMat = rightCameraRotationVector;
    CvMat rightCameraTranslationVectorMat= rightCameraTranslationVector;
    CvMat* leftCameraTriangulatedWorldPoints = cvCreateMat (1,3,CV_64FC1);
        
    CV_MAT_ELEM(*leftScreenPointsMat,double,0,0) = leftUndistorted.x;
    CV_MAT_ELEM(*leftScreenPointsMat,double,0,1) = leftUndistorted.y;
    CV_MAT_ELEM(*rightScreenPointsMat,double,0,0) = rightUndistorted.x;
    CV_MAT_ELEM(*rightScreenPointsMat,double,0,1) = rightUndistorted.y;

    mitk::CStyleTriangulatePointPairsUsingSVD(
      *leftScreenPointsMat,
      *rightScreenPointsMat,
      leftCameraIntrinsicMat,
      leftCameraRotationVectorMat,
      leftCameraTranslationVectorMat,
      rightCameraIntrinsicMat,
      rightCameraRotationVectorMat,
      rightCameraTranslationVectorMat,
      *leftCameraTriangulatedWorldPoints);

    cv::Point3d triangulatedGS;
    triangulatedGS.x = CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,0,0);
    triangulatedGS.y = CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,0,1);
    triangulatedGS.z = CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,0,2);
      
    po_leftLens.m_Points.push_back(triangulatedGS);

    cvReleaseMat (&leftScreenPointsMat);
    cvReleaseMat (&rightScreenPointsMat);
    cvReleaseMat (&rightScreenPointsMat);
  }
  return po_leftLens;
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::CalculateProjectionErrors (std::string outPrefix)
{
  if ( ! m_ProjectOK ) 
  {
    MITK_ERROR << "Attempted to run CalculateProjectionErrors, before running project(), no result.";
    return;
  }

  if ( ! m_GoldStandardPointsClassifiedOK )
  {
    ClassifyGoldStandardPoints ();
  }
  // for each point in the gold standard vectors m_LeftGoldStandardPoints
  // find the corresponding point in m_ProjectedPoints and calculate the projection 
  // error in pixels. We don't define what the point correspondence is, so 
  // maybe assume that the closest point is the match? Should be valid as long as the 
  // density of the projected points is significantly less than the expected errors
  // Then, estimate the mm error by taking the z measure from m_PointsInLeftLensCS
  // and projecting onto it.
  //
  for ( unsigned int i = 0 ; i < m_GoldStandardPoints.size() ; i ++ ) 
  {
    bool left = true;
    this->CalculateProjectionError( m_GoldStandardPoints[i]);
    this->CalculateReProjectionError ( m_GoldStandardPoints[i]);
  }

  std::ofstream lpout (std::string (outPrefix + "_leftProjection.errors").c_str());
  lpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_LeftProjectionErrors.size() ; i ++ )
  {
    lpout << m_LeftProjectionErrors[i] << std::endl;
  }
  cv::Point2d errorStdDev;
  cv::Point2d errorMean;
  errorMean = mitk::GetCentroid(m_LeftProjectionErrors, false, &errorStdDev);
  lpout << "#Mean Error     = " << errorMean << std::endl;
  lpout << "#StdDev         = " << errorStdDev << std::endl;
  double xrms = sqrt ( errorMean.x * errorMean.x + errorStdDev.x * errorStdDev.x );
  double yrms = sqrt ( errorMean.y * errorMean.y + errorStdDev.y * errorStdDev.y );
  double rms = sqrt ( xrms*xrms + yrms*yrms);
  lpout << "#rms            = " << xrms << ", " << yrms << ", " << rms << std::endl;
  errorMean = mitk::GetCentroid(m_LeftProjectionErrors, true, &errorStdDev);
  lpout << "#Ref Mean Error = " << errorMean << std::endl;
  lpout << "#Ref StdDev     = " << errorStdDev << std::endl; 
  xrms = sqrt ( errorMean.x * errorMean.x + errorStdDev.x * errorStdDev.x );
  yrms = sqrt ( errorMean.y * errorMean.y + errorStdDev.y * errorStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  lpout << "#Ref. rms       = " << xrms << ", " << yrms << ", " << rms << std::endl;
  lpout.close();

  std::ofstream rpout (std::string (outPrefix + "_rightProjection.errors").c_str());
  rpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_RightProjectionErrors.size() ; i ++ )
  {
    rpout << m_RightProjectionErrors[i] << std::endl;
  }
  errorMean = mitk::GetCentroid(m_RightProjectionErrors, false, &errorStdDev);
  rpout << "#Mean Error      = " << errorMean << std::endl;
  rpout << "#StdDev          = " << errorStdDev << std::endl; 
  xrms = sqrt ( errorMean.x * errorMean.x + errorStdDev.x * errorStdDev.x );
  yrms = sqrt ( errorMean.y * errorMean.y + errorStdDev.y * errorStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  rpout << "#rms             = " << xrms << ", " << yrms << ", " << rms << std::endl;
  errorMean = mitk::GetCentroid(m_RightProjectionErrors, true, &errorStdDev);
  rpout << "#Ref. Mean Error = " << errorMean << std::endl;
  rpout << "#Ref. StdDev     = " << errorStdDev << std::endl; 
  xrms = sqrt ( errorMean.x * errorMean.x + errorStdDev.x * errorStdDev.x );
  yrms = sqrt ( errorMean.y * errorMean.y + errorStdDev.y * errorStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  rpout << "#Ref. rms        = " << xrms << ", " << yrms << ", " << rms << std::endl;
  rpout.close();

  std::ofstream lrpout (std::string (outPrefix + "_leftReProjection.errors").c_str());
  lrpout << "#xmm ymm zmm" << std::endl;
  for ( unsigned int i = 0 ; i < m_LeftReProjectionErrors.size() ; i ++ )
  {
    lrpout << m_LeftReProjectionErrors[i] << std::endl;
  }
  cv::Point3d error3dStdDev;
  cv::Point3d error3dMean;
  error3dMean = mitk::GetCentroid(m_LeftReProjectionErrors, false, &error3dStdDev);
  lrpout << "#Mean Error      = " << error3dMean << std::endl;
  lrpout << "#StdDev          = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  lrpout << "#rms             = " << xrms << ", " << yrms << ", " << rms << std::endl;
  error3dMean = mitk::GetCentroid(m_LeftReProjectionErrors, true, &error3dStdDev);
  lrpout << "#Ref. Mean Error = " << error3dMean << std::endl;
  lrpout << "#Ref. StdDev     = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  lrpout << "#Ref. rms        = " << xrms << ", " << yrms << ", " << rms << std::endl;
  lrpout.close();

  std::ofstream rrpout (std::string (outPrefix + "_rightReProjection.errors").c_str());
  rrpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_RightReProjectionErrors.size() ; i ++ )
  {
    rrpout << m_RightReProjectionErrors[i] << std::endl;
  }
  error3dMean = mitk::GetCentroid(m_RightReProjectionErrors, false, &error3dStdDev);
  rrpout << "#Mean Error      = " << error3dMean << std::endl;
  rrpout << "#StdDev          = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  rrpout << "#rms             = " << xrms << ", " << yrms << ", " << rms << std::endl;
  error3dMean = mitk::GetCentroid(m_RightReProjectionErrors, true, &error3dStdDev);
  rrpout << "#Ref. Mean Error = " << error3dMean << std::endl;
  rrpout << "#Ref. StdDev     = " << error3dStdDev << std::endl; 
  xrms = sqrt ( error3dMean.x * error3dMean.x + error3dStdDev.x * error3dStdDev.x );
  yrms = sqrt ( error3dMean.y * error3dMean.y + error3dStdDev.y * error3dStdDev.y );
  rms = sqrt ( xrms*xrms + yrms*yrms);
  rrpout << "#Ref. rms        = " << xrms << ", " << yrms << ", " << rms << std::endl;
  rrpout.close();

}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::CalculateReProjectionError ( mitk::PickedObject GSPoint )
{
  mitk::PickedObject toMatch = GSPoint.CopyByHeader();
  toMatch.m_Channel = "left_lens";
  mitk::PickedObject matchingObject = GetMatchingPickedObject ( toMatch, *m_PointsInLeftLensCS[GSPoint.m_FrameNumber] );
  assert ( matchingObject.m_FrameNumber == GSPoint.m_FrameNumber );

  if ( GSPoint.m_Channel != "left" )
  {
    //transform points from left right lens
    for ( unsigned int i = 0 ; i < matchingObject.m_Points.size() ; i ++ ) 
    {

      cv::Mat m1 = cvCreateMat(3,1,CV_64FC1);
      m1.at<double>(0,0) = matchingObject.m_Points[i].x;
      m1.at<double>(1,0) = matchingObject.m_Points[i].y;
      m1.at<double>(2,0) = matchingObject.m_Points[i].z;

      m1 = m_RightToLeftRotationMatrix->inv() * m1 - *m_RightToLeftTranslationVector;
    
      matchingObject.m_Points[i].x = m1.at<double>(0,0);
      matchingObject.m_Points[i].y = m1.at<double>(1,0);
      matchingObject.m_Points[i].z = m1.at<double>(2,0);
    }
  }
  
  mitk::PickedObject undistortedObject = UndistortPickedObject ( GSPoint );
  //do a reprojectPickedobject function as well. At the moment I think we can only do 
  //reprojection for points. I guess what's needed is for the each point in the gold standard 
  //we find the nearest point on the projected model line (which generally won't be on a vertex
  mitk::PickedObject reprojectedObject = ReprojectPickedObject ( undistortedObject, matchingObject );
  
  reprojectedObject.m_Channel = "left_lens";
  if ( reprojectedObject.m_IsLine ) 
  {
    MITK_ERROR << "I can't do reprojection errors for lines yet";
    return;
  }

  cv::Point3d reprojectionError;
  reprojectedObject.DistanceTo ( matchingObject, reprojectionError, m_AllowableTimingError);
  if ( GSPoint.m_Channel != "left" )
  {
    m_LeftReProjectionErrors.push_back (reprojectionError);
  }
  else
  {
    m_RightReProjectionErrors.push_back (reprojectionError);
  }
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::CalculateProjectionError ( mitk::PickedObject GSPoint )
{
  mitk::PickedObject matchingObject = GetMatchingPickedObject ( GSPoint, *m_ProjectedPointLists[GSPoint.m_FrameNumber] );
  assert ( matchingObject.m_FrameNumber == GSPoint.m_FrameNumber );

  cv::Point3d projectionError;
  GSPoint.DistanceTo(matchingObject, projectionError, m_AllowableTimingError );
  if ( GSPoint.m_Channel == "left" ) 
  {
    m_LeftProjectionErrors.push_back(cv::Point2d ( projectionError.x, projectionError.y));
  }
  else
  {
    m_RightProjectionErrors.push_back(cv::Point2d ( projectionError.x, projectionError.y));
  }
}

//-----------------------------------------------------------------------------
bool ProjectPointsOnStereoVideo::FindNearestScreenPoint ( mitk::PickedObject& GSPickedObject )
{
  //we're assuming that  m_ClassifierProjectedPoints is in order of frames, better check this
  assert ( m_ProjectedPointLists[GSPickedObject.m_FrameNumber].IsNotNull() );
  assert ( GSPickedObject.m_FrameNumber ==  m_ProjectedPointLists[GSPickedObject.m_FrameNumber]->GetFrameNumber() );
  //let's check the timing errors while we're here
  long long timingError = static_cast<long long> ( GSPickedObject.m_TimeStamp ) -  static_cast <long long> (m_ProjectedPointLists[GSPickedObject.m_FrameNumber]->GetTimeStamp() ) ;
  if ( abs ( timingError > m_AllowableTimingError ) )
  {
    MITK_WARN << "Rejecting gold standard points at frame " << GSPickedObject.m_FrameNumber << " due to high timing error = " << timingError;
    return false;
  }

  if ( GSPickedObject.m_Id != -1 )
  {
    //don't need to do anything
    return true;
  }
  
  MITK_INFO << "Finding nearest screen point in frame " <<  GSPickedObject.m_FrameNumber;
  assert ( m_ClassifierProjectedPointLists[GSPickedObject.m_FrameNumber].IsNotNull() );
  assert ( GSPickedObject.m_FrameNumber ==  m_ClassifierProjectedPointLists[GSPickedObject.m_FrameNumber]->GetFrameNumber() );
  std::vector<mitk::PickedObject> ClassifierPoints = m_ClassifierProjectedPointLists[GSPickedObject.m_FrameNumber]->GetPickedObjects();

  double minRatio;
  mitk::PickedObject nearestObject = mitk::FindNearestPickedObject( GSPickedObject , ClassifierPoints , &minRatio );
  if ( minRatio < m_AllowablePointMatchingRatio || boost::math::isinf ( minRatio ) )
  {
    MITK_WARN << "Ambiguous object match or infinite match Ratio at frame " << GSPickedObject.m_FrameNumber << " discarding object from gold standard vector";
    return false;
  }

  GSPickedObject.m_Id = nearestObject.m_Id;
  return true;
}

//-----------------------------------------------------------------------------
mitk::PickedObject ProjectPointsOnStereoVideo::GetMatchingPickedObject ( const mitk::PickedObject& PickedObject ,
    const mitk::PickedPointList& PointList )
{
  assert ( PickedObject.m_FrameNumber ==  PointList.GetFrameNumber() );

  if ( PickedObject.m_Id == -1 )
  {
    MITK_ERROR << "Called get matching picked object with unclassified picked object, error";
    return mitk::PickedObject();
  }
  
  unsigned int matches = 0;
  mitk::PickedObject match;
  for ( unsigned int i = 0 ; i < PointList.GetListSize() ; i ++ )
  {
    if ( PickedObject.HeadersMatch ( PointList.GetPickedObjects()[i], m_AllowableTimingError ) )
    {
      match =  PointList.GetPickedObjects()[i];
      matches++;
    }
  }
  if ( matches == 0 )
  {
    MITK_ERROR << "Called get matching picked object but got not matches.";
  }
  if ( matches > 1 )
  {
    MITK_ERROR << "Called get matching picked object but multiple matches " << matches;
  }

  return match;
}

//-----------------------------------------------------------------------------
mitk::PickedObject ProjectPointsOnStereoVideo::UndistortPickedObject ( const mitk::PickedObject& po )
{
  assert ( po.m_Channel == "left" || po.m_Channel == "right" );
  mitk::PickedObject undistortedObject = po.CopyByHeader();
  bool cropUndistortedPointsToScreen = true;
  double cropValue = std::numeric_limits<double>::quiet_NaN();
  for ( unsigned int i = 0 ; i < po.m_Points.size () ; i ++ ) 
  {
    assert ( po.m_Points[i].z == 0 );
    cv::Point2d in = cv::Point2d ( po.m_Points[i].x, po.m_Points[i].y);
    cv::Point2d out;
    if ( po.m_Channel == "left" )
    {
      mitk::UndistortPoint (in, *m_LeftIntrinsicMatrix, 
        *m_LeftDistortionVector, out,
        cropUndistortedPointsToScreen , 
        0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    }
    else
    {
      mitk::UndistortPoint (in, *m_RightIntrinsicMatrix, 
        *m_RightDistortionVector, out,
        cropUndistortedPointsToScreen , 
        0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    }
    undistortedObject.m_Points.push_back(cv::Point3d ( out.x, out.y, 0.0 ));
  }

  return undistortedObject;

}

//-----------------------------------------------------------------------------
mitk::PickedObject ProjectPointsOnStereoVideo::ReprojectPickedObject ( const mitk::PickedObject& po, const mitk::PickedObject& reference )
{
  assert ( po.m_Channel == "left" || po.m_Channel == "right" );
  mitk::PickedObject reprojectedObject = po.CopyByHeader();

  //for a line we reproject to the plane through the centroid of the reference line. This is probably a good approximation
  double depth = mitk::GetCentroid ( reference.m_Points ).z;
  for ( unsigned int i = 0 ; i < po.m_Points.size () ; i ++ ) 
  {
    assert ( po.m_Points[i].z == 0 );
    cv::Point2d in = cv::Point2d ( po.m_Points[i].x, po.m_Points[i].y);
    cv::Point3d out;
    if ( po.m_Channel == "left" ) 
    {
      out = mitk::ReProjectPoint (in , *m_LeftIntrinsicMatrix);
    }
    else
    {
      out= mitk::ReProjectPoint ( in , *m_RightIntrinsicMatrix);
    }
    out.x *= depth;
    out.y *= depth;
    out.z *= depth;
    reprojectedObject.m_Points.push_back(out);
  }
  return reprojectedObject;
}

//-----------------------------------------------------------------------------
mitk::PickedPointList::Pointer  ProjectPointsOnStereoVideo::ProjectPickedPointList ( const mitk::PickedPointList::Pointer pl_leftLens, 
    const double& screenBuffer )
{
  //these should be projected to the right or left lens depending on the frame number, even for left, odd for right
  assert ( pl_leftLens->GetChannel() == "left_lens" );

  mitk::PickedPointList::Pointer projected_pl = pl_leftLens->CopyByHeader();

  assert (m_LeftGSFramesAreEven); //I'm not sure what would happen here if this wasn't the case

  cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_64FC1);
        leftCameraPositionToFocalPointUnitVector.at<double>(0,0)=0.0;
        leftCameraPositionToFocalPointUnitVector.at<double>(0,1)=0.0;
        leftCameraPositionToFocalPointUnitVector.at<double>(0,2)=1.0;
    
  bool cropUndistortedPointsToScreen = true;
  double cropValue = std::numeric_limits<double>::infinity();

  std::vector < mitk::PickedObject > pickedObjects = pl_leftLens->GetPickedObjects();
  std::vector < mitk::PickedObject > projectedObjects;
  for ( unsigned int i = 0 ; i < pickedObjects.size() ; i ++ ) 
  {
    //project onto screen
    mitk::PickedObject projectedObject = pickedObjects[i].CopyByHeader();
    CvMat* outputLeftCameraWorldPointsIn3D = NULL;
    CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
    CvMat* output2DPointsLeft = NULL ;
    CvMat* output2DPointsRight = NULL;
      
    cv::Mat leftCameraWorldPoints = cv::Mat (pickedObjects[i].m_Points.size(),3,CV_64FC1);
    cv::Mat leftCameraWorldNormals = cv::Mat (pickedObjects[i].m_Points.size(),3,CV_64FC1);
      
    for ( unsigned int j = 0 ; j < pickedObjects[i].m_Points.size() ; j ++ ) 
    {
      leftCameraWorldPoints.at<double>(j,0) = pickedObjects[i].m_Points[j].x;
      leftCameraWorldPoints.at<double>(j,1) = pickedObjects[i].m_Points[j].y;
      leftCameraWorldPoints.at<double>(j,2) = pickedObjects[i].m_Points[j].z;
      leftCameraWorldNormals.at<double>(j,0) = 0.0;
      leftCameraWorldNormals.at<double>(j,1) = 0.0;
      leftCameraWorldNormals.at<double>(j,2) = -1.0;
    }
    //this isn't the most efficient way of doing it but it is consistent with previous implementation 
    mitk::ProjectVisible3DWorldPointsToStereo2D
      ( leftCameraWorldPoints,leftCameraWorldNormals,
        leftCameraPositionToFocalPointUnitVector,
        *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,
        *m_RightIntrinsicMatrix,*m_RightDistortionVector,
        *m_RightToLeftRotationMatrix,*m_RightToLeftTranslationVector,
        outputLeftCameraWorldPointsIn3D,
        outputLeftCameraWorldNormalsIn3D,
        output2DPointsLeft,
        output2DPointsRight,
        cropUndistortedPointsToScreen , 
        0.0 - m_ProjectorScreenBuffer, m_VideoWidth + m_ProjectorScreenBuffer, 
        0.0 - m_ProjectorScreenBuffer, m_VideoHeight + m_ProjectorScreenBuffer,
        cropValue);
        
    if ( pl_leftLens->GetFrameNumber ()  % 2 == 0 )
    {
      for ( unsigned int j = 0 ; j < pickedObjects[i].m_Points.size() ; j ++ ) 
      {
        projectedObject.m_Points.push_back ( cv::Point3d ( CV_MAT_ELEM(*output2DPointsLeft,double,j,0), CV_MAT_ELEM(*output2DPointsLeft,double,j,1), 0.0 ) );
      }
      projectedObject.m_Channel = "left";
    }
    else 
    {
      for ( unsigned int j = 0 ; j < pickedObjects[i].m_Points.size() ; j ++ ) 
      {
        projectedObject.m_Points.push_back ( cv::Point3d ( CV_MAT_ELEM(*output2DPointsRight,double,j,0), CV_MAT_ELEM(*output2DPointsRight,double,j,1), 0.0 ) );
      }
      projectedObject.m_Channel = "right";
    }
    projectedObjects.push_back (projectedObject);

    cvReleaseMat(&outputLeftCameraWorldPointsIn3D);
    cvReleaseMat(&outputLeftCameraWorldNormalsIn3D);
    cvReleaseMat(&output2DPointsLeft);
    cvReleaseMat(&output2DPointsRight);
  }
  projected_pl->SetPickedObjects ( projectedObjects );
  if ( pl_leftLens->GetFrameNumber ()  % 2 == 0 )
  {
    projected_pl->SetChannel ( "left" );
  }
  else
  {
    projected_pl->SetChannel ( "right" );
  }

  return projected_pl;
}

//-----------------------------------------------------------------------------
mitk::PickedPointList::Pointer  ProjectPointsOnStereoVideo::TransformPickedPointListToLeftLens ( const mitk::PickedPointList::Pointer pl_world, const cv::Mat& transform, const unsigned long long& timestamp, const int& framenumber)
{
  //these should be projected to the right or left lens depending on the frame number, even for left, odd for right
  assert ( pl_world->GetChannel() == "world" );
 
  mitk::PickedPointList::Pointer transformedList = pl_world->CopyByHeader();
  transformedList->SetTimeStamp ( timestamp );
  transformedList->SetFrameNumber (framenumber);
  
  std::vector < mitk::PickedObject > pickedObjects = pl_world->GetPickedObjects();
  
  for ( unsigned int i = 0 ; i < pickedObjects.size() ; i ++ ) 
  {
      pickedObjects[i].m_Points = transform * pickedObjects[i].m_Points;
      pickedObjects[i].m_TimeStamp = timestamp;
      pickedObjects[i].m_FrameNumber = framenumber;
      pickedObjects[i].m_Channel = "left_lens";
  }
  
  transformedList->SetPickedObjects (pickedObjects);
  transformedList->SetChannel ("left_lens");
  return transformedList;
}


//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::ClassifyGoldStandardPoints ()
{
  assert (m_ProjectOK);
  std::sort ( m_GoldStandardPoints.begin(), m_GoldStandardPoints.end());
  unsigned int startsize = m_GoldStandardPoints.size();
  MITK_INFO << "MITK classifing " << startsize << " gold standard points ";
  for ( std::vector<mitk::PickedObject>::iterator it = m_GoldStandardPoints.end() - 1  ; it >= m_GoldStandardPoints.begin() ; --it ) 
  {
    if ( ! this->FindNearestScreenPoint ( *it ) )
    {
      m_GoldStandardPoints.erase ( it );
    }
  }
  MITK_INFO << "Removed " << startsize - m_GoldStandardPoints.size() << " objects from gold standard vector, " <<  m_GoldStandardPoints.size() << " objects left.";
 
  m_GoldStandardPointsClassifiedOK = true;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::AppendWorldPoints ( 
    std::vector < mitk::WorldPoint > points )
{
  if ( m_WorldPoints.IsNull() ) 
  {
    m_WorldPoints = mitk::PickedPointList::New();
    m_WorldPoints->SetChannel ("world");
  }
  m_WorldPoints->SetInLineMode(false);
  m_WorldPoints->SetInOrderedMode(true);
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_WorldPoints->AddPoint ( points[i].m_Point, points[i].m_Scalar );
  }
  m_ProjectOK = false;
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::AppendClassifierWorldPoints ( 
    std::vector < mitk::WorldPoint > points )
{
  if ( m_ClassifierWorldPoints.IsNull() )
  {
    m_ClassifierWorldPoints = mitk::PickedPointList::New();
    m_ClassifierWorldPoints->SetChannel ("world");
  }
  m_ClassifierWorldPoints->SetInLineMode(false);
  m_ClassifierWorldPoints->SetInOrderedMode(true);
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_ClassifierWorldPoints->AddPoint (points[i].m_Point, points[i].m_Scalar);
  }
  m_ProjectOK = false;
}

//-----------------------------------------------------------------------------
std::vector <mitk::PickedPointList::Pointer> ProjectPointsOnStereoVideo::GetPointsInLeftLensCS ()
{
  return m_PointsInLeftLensCS;
}

//-----------------------------------------------------------------------------
std::vector <mitk::ProjectedPointPairsWithTimingError> ProjectPointsOnStereoVideo::GetProjectedPoints ()
{
   MITK_ERROR << "get projected points is  broke";
   assert (false);
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::AppendWorldPointsByTriangulation
    (std::vector< mitk::ProjectedPointPair > onScreenPointPairs,
     std::vector< unsigned int > framenumber  , 
     mitk::VideoTrackerMatching::Pointer trackerMatcher, 
     std::vector<double> * perturbation)
{
  assert ( framenumber.size() == onScreenPointPairs.size() );

  if ( m_WorldPoints.IsNull() ) 
  {
    m_WorldPoints = mitk::PickedPointList::New();
    m_WorldPoints->SetChannel ("world");
  }
  if ( ! trackerMatcher->IsReady () ) 
  {
    MITK_ERROR << "Attempted to triangulate points without tracking matrices.";
    return;
  }
  
  std::vector < mitk::WorldPoint > leftLensPoints = 
    mitk::Triangulate ( onScreenPointPairs, 
      *m_LeftIntrinsicMatrix,
      *m_LeftDistortionVector,
      *m_RightIntrinsicMatrix,
      *m_RightDistortionVector,
      *m_RightToLeftRotationMatrix,
      *m_RightToLeftTranslationVector,
      true,
      0.0, m_VideoWidth, 0.0 , m_VideoHeight, 
      std::numeric_limits<double>::quiet_NaN());

    mitk::WorldPoint point;
    unsigned int wpSize=m_WorldPoints->GetListSize();
  
    for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
    {
      point = leftLensPoints[i];
      long long timingError;
      point =  trackerMatcher->GetCameraTrackingMatrix(
          framenumber[i] , &timingError , m_TrackerIndex, perturbation, m_ReferenceIndex) * point;
    if ( abs(timingError) < m_AllowableTimingError )
    {
      m_WorldPoints->AddPoint ( point.m_Point, point.m_Scalar );
      MITK_INFO << framenumber[i] << " " << onScreenPointPairs[i].m_Left << ","
        << onScreenPointPairs[i].m_Right << " => " << point.m_Point << " => " << m_WorldPoints->GetPickedObjects()[i+wpSize].m_Points[0];
    }
    else
    {
      MITK_WARN << framenumber[i] << "Point rejected due to excessive timing error: " << timingError << " > " << m_AllowableTimingError;
    }

  }
  m_ProjectOK = false;
}

void ProjectPointsOnStereoVideo::ProjectAxes()
{
  cv::Mat leftCameraAxesPoints = cv::Mat (4,3,CV_64FC1);
  cv::Mat leftCameraAxesNormals = cv::Mat (4,3,CV_64FC1);
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    int j;
    for ( j = 0 ; j < 2 ; j ++ ) 
    {
      leftCameraAxesPoints.at<double>(i,j) = 0.0;
      leftCameraAxesNormals.at<double>(i,j) = 0.0;
    }
      leftCameraAxesPoints.at<double>(i,j) = 100.0;
      leftCameraAxesNormals.at<double>(i,j) = -1.0;
  }
  leftCameraAxesPoints.at<double>(1,0) = 100;
  leftCameraAxesPoints.at<double>(2,1) = 100;
  leftCameraAxesPoints.at<double>(3,2) = 200;
      
  CvMat* outputLeftCameraAxesPointsIn3D = NULL;
  CvMat* outputLeftCameraAxesNormalsIn3D = NULL ;
  CvMat* output2DAxesPointsLeft = NULL ;
  CvMat* output2DAxesPointsRight = NULL;
  
  cv::Mat CameraUnitVector = cv::Mat(1,3,CV_64FC1);
  CameraUnitVector.at<double>(0,0)=0;
  CameraUnitVector.at<double>(0,1)=0;
  CameraUnitVector.at<double>(0,2)=1.0;
  mitk::ProjectVisible3DWorldPointsToStereo2D
    ( leftCameraAxesPoints,leftCameraAxesNormals,
        CameraUnitVector,
        *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,
        *m_RightIntrinsicMatrix,*m_RightDistortionVector,
        *m_RightToLeftRotationMatrix,*m_RightToLeftTranslationVector,
        outputLeftCameraAxesPointsIn3D,
        outputLeftCameraAxesNormalsIn3D,
        output2DAxesPointsLeft,
        output2DAxesPointsRight);
  
  MITK_INFO << leftCameraAxesPoints;
  for ( unsigned int i = 0 ; i < 4 ; i ++ ) 
  {
    MITK_INFO << i;
    mitk::ProjectedPointPair pointPair;
    pointPair.m_Left = cv::Point2d(CV_MAT_ELEM(*output2DAxesPointsLeft,double,i,0),CV_MAT_ELEM(*output2DAxesPointsLeft,double,i,1));
    pointPair.m_Right = cv::Point2d(CV_MAT_ELEM(*output2DAxesPointsRight,double,i,0),CV_MAT_ELEM(*output2DAxesPointsRight,double,i,1));
    MITK_INFO << "Left" << pointPair.m_Left << "Right" << pointPair.m_Right;

    m_ScreenAxesPoints.m_Points.push_back(pointPair);
  }
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::ClearWorldPoints()
{
  m_WorldPoints->ClearList();
  m_ProjectOK = false;
}

} // end namespace
