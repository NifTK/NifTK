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
#include <mitkPointSetWriter.h>
#include <cv.h>
//#include <opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include <niftkFileHelper.h>

#include <boost/filesystem.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::ProjectPointsOnStereoVideo()
: m_Visualise(false)
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_Directory("")
, m_VideoOutPrefix("")
, m_TriangulatedPointsOutName("")
, m_TrackerIndex(0)
, m_ReferenceIndex(-1)
, m_DrawLines(false)
, m_InitOK(false)
, m_ProjectOK(false)
, m_DrawAxes(false)
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
  ProjectAxes(); 
  
  if ( m_Visualise || m_SaveVideo ) 
  {
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
    if ( m_SaveVideo )
    {
      cv::Size S = cv::Size((int) m_VideoWidth/2.0, (int) m_VideoHeight );
      double fps = (double)cvGetCaptureProperty (m_Capture, CV_CAP_PROP_FPS);
      double halfFPS = fps/2.0;
      m_LeftWriter =cvCreateVideoWriter(std::string(m_VideoOutPrefix + "leftchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),halfFPS,S, true);
      m_RightWriter =cvCreateVideoWriter(std::string(m_VideoOutPrefix + "rightchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),halfFPS,S, true);
    }
  }

  m_InitOK = true;
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
void ProjectPointsOnStereoVideo::SetSaveVideo ( bool savevideo, std::string prefix )
{
  if ( m_InitOK ) 
  {
    MITK_WARN << "Changing save video  state after initialisation, will need to re-initialise";
  }
  m_SaveVideo = savevideo;
  m_VideoOutPrefix = prefix;

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
    
  m_ProjectOK = false;
  m_ProjectedPoints.clear();
  m_PointsInLeftLensCS.clear();
  m_ClassifierProjectedPoints.clear();
  if ( static_cast<int>(m_WorldPoints.size()) < m_MaxGoldStandardIndex ) 
  {
    MITK_INFO << "Filling world points with dummy data to enable triangulation";
    mitk::WorldPoint emptyWorldPoint;

    for ( int i = m_WorldPoints.size() ; i <= m_MaxGoldStandardIndex ; i ++ )
    {
      m_WorldPoints.push_back(emptyWorldPoint);
    }
  }
  if (  m_WorldPoints.size() == 0 )
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
        cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
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
      
      m_WorldToLeftCameraMatrices.push_back(WorldToLeftCamera);
      mitk::WorldPointsWithTimingError pointsInLeftLensCS = 
        mitk::WorldPointsWithTimingError ( WorldToLeftCamera * m_WorldPoints, timingError); 
      
      mitk::WorldPointsWithTimingError classifierPointsInLeftLensCS = 
        mitk::WorldPointsWithTimingError ( WorldToLeftCamera * m_ClassifierWorldPoints, timingError);

      m_PointsInLeftLensCS.push_back (pointsInLeftLensCS); 

      //project onto screen
      CvMat* outputLeftCameraWorldPointsIn3D = NULL;
      CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
      CvMat* output2DPointsLeft = NULL ;
      CvMat* output2DPointsRight = NULL;
      
      cv::Mat leftCameraWorldPoints = cv::Mat (pointsInLeftLensCS.m_Points.size(),3,CV_64FC1);
      cv::Mat leftCameraWorldNormals = cv::Mat (pointsInLeftLensCS.m_Points.size(),3,CV_64FC1);
      
      CvMat* classifierOutputLeftCameraWorldPointsIn3D = NULL;
      CvMat* classifierOutputLeftCameraWorldNormalsIn3D = NULL ;
      CvMat* classifierOutput2DPointsLeft = NULL ;
      CvMat* classifierOutput2DPointsRight = NULL;
      
      cv::Mat classifierLeftCameraWorldPoints = cv::Mat (classifierPointsInLeftLensCS.m_Points.size(),3,CV_64FC1);
      cv::Mat classifierLeftCameraWorldNormals = cv::Mat (classifierPointsInLeftLensCS.m_Points.size(),3,CV_64FC1);
      
      for ( unsigned int i = 0 ; i < pointsInLeftLensCS.m_Points.size() ; i ++ ) 
      {
        leftCameraWorldPoints.at<double>(i,0) = pointsInLeftLensCS.m_Points[i].m_Point.x;
        leftCameraWorldPoints.at<double>(i,1) = pointsInLeftLensCS.m_Points[i].m_Point.y;
        leftCameraWorldPoints.at<double>(i,2) = pointsInLeftLensCS.m_Points[i].m_Point.z;
        leftCameraWorldNormals.at<double>(i,0) = 0.0;
        leftCameraWorldNormals.at<double>(i,1) = 0.0;
        leftCameraWorldNormals.at<double>(i,2) = -1.0;
      }
      for ( unsigned int i = 0 ; i < classifierPointsInLeftLensCS.m_Points.size() ; i ++ ) 
      {
        classifierLeftCameraWorldPoints.at<double>(i,0) = classifierPointsInLeftLensCS.m_Points[i].m_Point.x;
        classifierLeftCameraWorldPoints.at<double>(i,1) = classifierPointsInLeftLensCS.m_Points[i].m_Point.y;
        classifierLeftCameraWorldPoints.at<double>(i,2) = classifierPointsInLeftLensCS.m_Points[i].m_Point.z;
        classifierLeftCameraWorldNormals.at<double>(i,0) = 0.0;
        classifierLeftCameraWorldNormals.at<double>(i,1) = 0.0;
        classifierLeftCameraWorldNormals.at<double>(i,2) = -1.0;
      }
     
      cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_64FC1);
      leftCameraPositionToFocalPointUnitVector.at<double>(0,0)=0.0;
      leftCameraPositionToFocalPointUnitVector.at<double>(0,1)=0.0;
      leftCameraPositionToFocalPointUnitVector.at<double>(0,2)=1.0;
    
      bool cropUndistortedPointsToScreen = true;
      double cropValue = std::numeric_limits<double>::infinity();
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
      
      mitk::ProjectVisible3DWorldPointsToStereo2D
        ( classifierLeftCameraWorldPoints,classifierLeftCameraWorldNormals,
          leftCameraPositionToFocalPointUnitVector,
          *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,
          *m_RightIntrinsicMatrix,*m_RightDistortionVector,
          *m_RightToLeftRotationMatrix,*m_RightToLeftTranslationVector,
          classifierOutputLeftCameraWorldPointsIn3D,
          classifierOutputLeftCameraWorldNormalsIn3D,
          classifierOutput2DPointsLeft,
          classifierOutput2DPointsRight,
          cropUndistortedPointsToScreen , 
          0.0 - m_ClassifierScreenBuffer, m_VideoWidth + m_ClassifierScreenBuffer, 
          0.0 - m_ClassifierScreenBuffer, m_VideoHeight + m_ClassifierScreenBuffer,
          cropValue);

      std::vector < mitk::ProjectedPointPair > screenPoints;
      std::vector < mitk::ProjectedPointPair > classifierScreenPoints;
      
      for ( unsigned int i = 0 ; i < pointsInLeftLensCS.m_Points.size() ; i ++ ) 
      {
        mitk::ProjectedPointPair pointPair;
        pointPair.m_Left = cv::Point2d(CV_MAT_ELEM(*output2DPointsLeft,double,i,0),CV_MAT_ELEM(*output2DPointsLeft,double,i,1));
        pointPair.m_Right = cv::Point2d(CV_MAT_ELEM(*output2DPointsRight,double,i,0),CV_MAT_ELEM(*output2DPointsRight,double,i,1));
        screenPoints.push_back(pointPair);
      }
      m_ProjectedPoints.push_back(mitk::ProjectedPointPairsWithTimingError ( screenPoints, timingError ));
      
      for ( unsigned int i = 0 ; i < classifierPointsInLeftLensCS.m_Points.size() ; i ++ ) 
      {
        mitk::ProjectedPointPair pointPair;
        pointPair.m_Left = cv::Point2d(CV_MAT_ELEM(*classifierOutput2DPointsLeft,double,i,0),CV_MAT_ELEM(*classifierOutput2DPointsLeft,double,i,1));
        pointPair.m_Right = cv::Point2d(CV_MAT_ELEM(*classifierOutput2DPointsRight,double,i,0),CV_MAT_ELEM(*classifierOutput2DPointsRight,double,i,1));
        classifierScreenPoints.push_back(pointPair);
      }
      m_ClassifierProjectedPoints.push_back(mitk::ProjectedPointPairsWithTimingError ( classifierScreenPoints, timingError));
      
      //de-allocate the matrices    
      cvReleaseMat(&outputLeftCameraWorldPointsIn3D);
      cvReleaseMat(&outputLeftCameraWorldNormalsIn3D);
      cvReleaseMat(&output2DPointsLeft);
      cvReleaseMat(&output2DPointsRight);
      cvReleaseMat(&classifierOutputLeftCameraWorldPointsIn3D);
      cvReleaseMat(&classifierOutputLeftCameraWorldNormalsIn3D);
      cvReleaseMat(&classifierOutput2DPointsLeft);
      cvReleaseMat(&classifierOutput2DPointsRight);

      if ( m_Visualise || m_SaveVideo ) 
      {
        cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
        for ( unsigned thing = 0 ; thing < m_WorldPoints.size() ; thing ++ )
        MITK_INFO << framenumber << " " << m_WorldPoints[thing].m_Point << " " << pointsInLeftLensCS.m_Points[thing].m_Scalar << " => " << screenPoints[thing].m_Left << screenPoints[thing].m_Right;
        if ( drawProjection )
        {
          if ( ! m_DrawLines ) 
          {
            if ( framenumber % 2 == 0 ) 
            {
              for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
              {
                cv::circle(videoImage, screenPoints[i].m_Left,10, pointsInLeftLensCS.m_Points[i].m_Scalar, 3, 8, 0 );
              }
            }
            else
            {
              for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
              {
                cv::circle(videoImage, screenPoints[i].m_Right,10, pointsInLeftLensCS.m_Points[i].m_Scalar, 3, 8, 0 );
              } 
            }
          }
          else 
          {
            if ( framenumber % 2 == 0 ) 
            {
              unsigned int i;
              for (i = 0 ; i < screenPoints.size()-1 ; i ++ ) 
              {
                cv::line(videoImage, screenPoints[i].m_Left,screenPoints[i+1].m_Left, pointsInLeftLensCS.m_Points[i].m_Scalar );
              }
              cv::line(videoImage, screenPoints[i].m_Left,screenPoints[0].m_Left, pointsInLeftLensCS.m_Points[i].m_Scalar );
            }
            else
            {
              unsigned int i;
              for ( i = 0 ; i < screenPoints.size()-1 ; i ++ ) 
              {
                cv::line(videoImage, screenPoints[i].m_Right,screenPoints[i+1].m_Right, pointsInLeftLensCS.m_Points[i].m_Scalar );
              } 
              cv::line(videoImage, screenPoints[i].m_Right,screenPoints[0].m_Right, pointsInLeftLensCS.m_Points[i].m_Scalar );
            }
          }
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
    std::vector < mitk::GoldStandardPoint > points )
{
  m_LeftGoldStandardPoints = points;
  int maxLeftGSIndex = -1;
  for ( unsigned int i = 0 ; i < m_LeftGoldStandardPoints.size() ; i ++ ) 
  {
    if ( m_LeftGoldStandardPoints[i].m_Index > maxLeftGSIndex ) 
    {
      maxLeftGSIndex =  m_LeftGoldStandardPoints[i].m_Index;
    }
    if ( m_LeftGoldStandardPoints[i].m_FrameNumber % 2 == 0 ) 
    {
      if ( ! m_LeftGSFramesAreEven ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points";
        exit(1);
      }
    }
    else
    {
      if ( ( i > 0 ) && ( m_LeftGSFramesAreEven ) ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the left gold standard points";
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
void ProjectPointsOnStereoVideo::SetRightGoldStandardPoints (
    std::vector < mitk::GoldStandardPoint > points )
{
  m_RightGoldStandardPoints = points;
  int maxRightGSIndex = -1;
   
  for ( unsigned int i = 0 ; i < m_RightGoldStandardPoints.size() ; i ++ ) 
  {
    if ( m_RightGoldStandardPoints[i].m_Index > maxRightGSIndex ) 
    {
      maxRightGSIndex =  m_RightGoldStandardPoints[i].m_Index;
    }
    if ( m_RightGoldStandardPoints[i].m_FrameNumber % 2 == 0 ) 
    {
      if ( ! m_RightGSFramesAreEven ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points";
        exit(1);
      }
    }
    else
    {
      if ( ( i > 0 ) && ( m_RightGSFramesAreEven ) ) 
      {
        MITK_ERROR << "Detected inconsistent frame numbering in the right gold standard points";
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
void ProjectPointsOnStereoVideo::CalculateTriangulationErrors (std::string outPrefix, 
    mitk::VideoTrackerMatching::Pointer trackerMatcher)
{
  if ( (! m_ProjectOK) && (m_MaxGoldStandardIndex == -1 ) ) 
  {
    MITK_ERROR << "Attempted to run CalculateTriangulateErrors, before running project(), no result.";
    return;
  }

  std::sort ( m_LeftGoldStandardPoints.begin(), m_LeftGoldStandardPoints.end());
  std::sort ( m_RightGoldStandardPoints.begin(), m_RightGoldStandardPoints.end());
  
  unsigned int leftGSIndex = 0;
  unsigned int rightGSIndex = 0;

  std::vector < std::vector < cv::Point3d > > classifiedPoints;

  for ( unsigned int i = 0 ; i < m_PointsInLeftLensCS[0].m_Points.size() ; i ++ )
  {
    std::vector < cv::Point3d > pointvector;
    classifiedPoints.push_back(pointvector);
  }

  while ( leftGSIndex < m_LeftGoldStandardPoints.size() && rightGSIndex < m_RightGoldStandardPoints.size() )
  {
    unsigned int frameNumber = m_LeftGoldStandardPoints[leftGSIndex].m_FrameNumber;
    std::vector < mitk::GoldStandardPoint > leftPoints;
    std::vector < mitk::GoldStandardPoint > rightPoints;
    std::vector < std::pair < unsigned int , std::pair < cv::Point2d , cv::Point2d > > > matchedPairs;
    
    while ( m_LeftGoldStandardPoints[leftGSIndex].m_FrameNumber == frameNumber && leftGSIndex < m_LeftGoldStandardPoints.size() ) 
    {
      leftPoints.push_back ( m_LeftGoldStandardPoints[leftGSIndex] );
      leftGSIndex ++;
    }
    while (  m_RightGoldStandardPoints[rightGSIndex].m_FrameNumber < ( frameNumber + m_RightGSFrameOffset ) && rightGSIndex < m_RightGoldStandardPoints.size() )  
    {
      rightGSIndex ++;
    }

    while ( m_RightGoldStandardPoints[rightGSIndex].m_FrameNumber == ( frameNumber + m_RightGSFrameOffset )  && rightGSIndex < m_RightGoldStandardPoints.size() )  
    {
      rightPoints.push_back ( m_RightGoldStandardPoints[rightGSIndex] );
      rightGSIndex ++;
    }
//check timing error here
    if ( abs (m_PointsInLeftLensCS[frameNumber].m_TimingError) < m_AllowableTimingError )
    {
      for ( unsigned int i = 0 ; i < leftPoints.size() ; i ++ ) 
      {
        unsigned int index;
        double minRatio;
        bool left = true;
        this->FindNearestScreenPoint (  leftPoints[i] , left, &minRatio, &index );
        if ( minRatio < m_AllowablePointMatchingRatio || boost::math::isinf (minRatio) ) 
        {
          MITK_WARN << "Ambiguous point match or infinite match Ratio at left frame " << frameNumber << " point " << i << " discarding point from triangulation  errors"; 
        }
        else 
        {
          left = false;
          for ( unsigned int j = 0 ; j < rightPoints.size() ; j ++ ) 
          {
            unsigned int rightIndex;
            this->FindNearestScreenPoint (   
              rightPoints[j] , left, &minRatio, &rightIndex );
            if ( minRatio < m_AllowablePointMatchingRatio || boost::math::isinf(minRatio) ) 
            {
              MITK_WARN << "Ambiguous point match or infinite match Ratio at right frame " << frameNumber << " point " << j << " discarding point from triangulation errors"; 
            }
            else
            {
              if ( rightIndex == index ) 
              {
                matchedPairs.push_back( std::pair < unsigned int , std::pair < cv::Point2d , cv::Point2d > >
                (index, std::pair <cv::Point2d, cv::Point2d> ( leftPoints[i].m_Point, rightPoints[j].m_Point )));
              }
            }
          }
        }
      }

      for ( unsigned int i = 0 ; i < matchedPairs.size() ; i ++ ) 
      {
        cv::Point2d leftUndistorted;
        bool cropUndistortedPointsToScreen = true;
        double cropValue = std::numeric_limits<double>::quiet_NaN();
        mitk::UndistortPoint(matchedPairs[i].second.first,
               *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,leftUndistorted,
               cropUndistortedPointsToScreen , 
               0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
        cv::Point2d rightUndistorted;
        mitk::UndistortPoint(matchedPairs[i].second.second,
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
      
        m_TriangulationErrors.push_back(triangulatedGS - 
            m_PointsInLeftLensCS[frameNumber].m_Points[matchedPairs[i].first].m_Point);
        
        cv::Mat leftCameraToWorld = trackerMatcher->GetCameraTrackingMatrix(frameNumber, NULL, m_TrackerIndex, NULL, m_ReferenceIndex);
        
        classifiedPoints[matchedPairs[i].first].push_back(leftCameraToWorld * triangulatedGS);
        cvReleaseMat (&leftScreenPointsMat);
        cvReleaseMat (&rightScreenPointsMat);
        cvReleaseMat (&rightScreenPointsMat);
      }
    }
    else 
    {
      MITK_WARN << "Rejecting triangulation error at frame " << frameNumber << " due to high timing error " << m_PointsInLeftLensCS[frameNumber].m_TimingError << " > "  << m_AllowableTimingError ;
    }
  }
  mitk::PointSet::Pointer triangulatedPoints = mitk::PointSet::New();
  for ( unsigned int i = 0 ; i < classifiedPoints.size() ; i ++ ) 
  {
    cv::Point3d centroid;
    cv::Point3d stdDev;
    centroid = mitk::GetCentroid (classifiedPoints[i],true, & stdDev);

    mitk::Point3D point;
    point[0] = centroid.x;
    point[1] = centroid.y;
    point[2] = centroid.z;
    triangulatedPoints->InsertPoint(i,point);
    MITK_INFO << "Point " << i << " triangulated mean " << centroid << " SD " << stdDev;
  }
  if ( m_TriangulatedPointsOutName != "" )
  {
    mitk::PointSetWriter::Pointer tpWriter = mitk::PointSetWriter::New();
    tpWriter->SetFileName(m_TriangulatedPointsOutName);
    tpWriter->SetInput( triangulatedPoints );
    tpWriter->Update();
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
void ProjectPointsOnStereoVideo::CalculateProjectionErrors (std::string outPrefix)
{
  if ( ! m_ProjectOK ) 
  {
    MITK_ERROR << "Attempted to run CalculateProjectionErrors, before running project(), no result.";
    return;
  }

  // for each point in the gold standard vectors m_LeftGoldStandardPoints
  // find the corresponding point in m_ProjectedPoints and calculate the projection 
  // error in pixels. We don't define what the point correspondence is, so 
  // maybe assume that the closest point is the match? Should be valid as long as the 
  // density of the projected points is significantly less than the expected errors
  // Then, estimate the mm error by taking the z measure from m_PointsInLeftLensCS
  // and projecting onto it.
  //
  for ( unsigned int i = 0 ; i < m_LeftGoldStandardPoints.size() ; i ++ ) 
  {
    bool left = true;
    this->CalculateProjectionError( m_LeftGoldStandardPoints[i], left);
    this->CalculateReProjectionError ( m_LeftGoldStandardPoints[i] , left);
  }
  for ( unsigned int i = 0 ; i < m_RightGoldStandardPoints.size() ; i ++ ) 
  {
    bool left = false;
    this->CalculateProjectionError( m_RightGoldStandardPoints[i], left);
    this->CalculateReProjectionError ( m_RightGoldStandardPoints[i], left);
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
void ProjectPointsOnStereoVideo::CalculateReProjectionError ( GoldStandardPoint GSPoint, bool left )
{
  unsigned int* index = new unsigned int;
  double minRatio;
  FindNearestScreenPoint ( GSPoint, left, &minRatio, index ) ;
  std::string side;
  if ( left ) 
  {
    side = " left ";
  }
  else
  {
    side = " right "; 
  }
  if ( abs (m_PointsInLeftLensCS[GSPoint.m_FrameNumber].m_TimingError) > m_AllowableTimingError ) 
  {
    MITK_WARN << "High timing error at " << side << " frame " << GSPoint.m_FrameNumber << " discarding point from re-projection errors";
    return;
  }

  if ( minRatio < m_AllowablePointMatchingRatio ) 
  {
    MITK_WARN << "Ambiguous point match at " << side  << "frame "  << GSPoint.m_FrameNumber << " discarding point from re-projection errors"; 
    return;
  }

  if ( boost::math::isinf(minRatio) ) 
  {
    MITK_WARN << "Infinite match ratio at " << side  << "frame "  << GSPoint.m_FrameNumber << " discarding point from re-projection errors"; 
    return;
  }


  cv::Point3d matchingPointInLensCS = m_PointsInLeftLensCS[GSPoint.m_FrameNumber].m_Points[*index].m_Point;

  if ( ! left )
  {
    cv::Mat m1 = cvCreateMat(3,1,CV_64FC1);
    m1.at<double>(0,0) = matchingPointInLensCS.x;
    m1.at<double>(1,0) = matchingPointInLensCS.y;
    m1.at<double>(2,0) = matchingPointInLensCS.z;

    m1 = m_RightToLeftRotationMatrix->inv() * m1 - *m_RightToLeftTranslationVector;
    
    matchingPointInLensCS.x = m1.at<double>(0,0);
    matchingPointInLensCS.y = m1.at<double>(1,0);
    matchingPointInLensCS.z = m1.at<double>(2,0);
  }
  
  cv::Point3d reProjectionGS;
  bool cropUndistortedPointsToScreen = true;
  double cropValue = std::numeric_limits<double>::quiet_NaN();
  if ( left ) 
  {
    cv::Point2d undistortedPoint;
    mitk::UndistortPoint (GSPoint.m_Point, *m_LeftIntrinsicMatrix, 
        *m_LeftDistortionVector, undistortedPoint,
        cropUndistortedPointsToScreen , 
        0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    reProjectionGS = mitk::ReProjectPoint (undistortedPoint , *m_LeftIntrinsicMatrix);
  }
  else
  {
    cv::Point2d undistortedPoint;
    mitk::UndistortPoint (GSPoint.m_Point, *m_RightIntrinsicMatrix, 
        *m_RightDistortionVector, undistortedPoint,
        cropUndistortedPointsToScreen , 
        0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    reProjectionGS = mitk::ReProjectPoint (undistortedPoint , *m_RightIntrinsicMatrix);
  }
  
  reProjectionGS.x *= matchingPointInLensCS.z;
  reProjectionGS.y *= matchingPointInLensCS.z;
  reProjectionGS.z *= matchingPointInLensCS.z;

  delete index;
  if ( left ) 
  {
    m_LeftReProjectionErrors.push_back (matchingPointInLensCS - reProjectionGS);
  }
  else
  {
    m_RightReProjectionErrors.push_back (matchingPointInLensCS - reProjectionGS);
  }

}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::CalculateProjectionError ( GoldStandardPoint GSPoint, bool left )
{
  double minRatio;
  cv::Point2d matchingPoint = FindNearestScreenPoint ( GSPoint, left, &minRatio ) ;
  std::string side;
  if ( left ) 
  {
    side = " left ";
  }
  else
  {
    side = " right "; 
  }
 
  if ( abs (m_PointsInLeftLensCS[GSPoint.m_FrameNumber].m_TimingError) > m_AllowableTimingError ) 
  {
    MITK_WARN << "High timing error at " << side << "  frame " << GSPoint.m_FrameNumber << " discarding point from projection errors";
    return;
  }


  if ( minRatio < m_AllowablePointMatchingRatio ) 
  {
    MITK_WARN << "Ambiguous point match at " << side << "frame " << GSPoint.m_FrameNumber << " discarding point from projection errors"; 
    return;
  }
  
  if ( boost::math::isinf(minRatio) ) 
  {
    MITK_WARN << "Infinite match ratio at " << side  << "frame "  << GSPoint.m_FrameNumber << " discarding point from projection errors"; 
    return;
  }


  if ( left ) 
  {
    m_LeftProjectionErrors.push_back(matchingPoint - GSPoint.m_Point);
  }
  else
  {
    m_RightProjectionErrors.push_back(matchingPoint - GSPoint.m_Point);
  }

}

//-----------------------------------------------------------------------------
cv::Point2d ProjectPointsOnStereoVideo::FindNearestScreenPoint ( GoldStandardPoint GSPoint, bool left , double* minRatio, unsigned int* index)
{
  if ( GSPoint.m_Index != -1 )
  {
    if ( index != NULL ) 
    {
      *index = GSPoint.m_Index;
    }
    if ( minRatio != NULL ) 
    {
      *minRatio = m_AllowablePointMatchingRatio + 1.0;
    }
    if ( left )
    {
      return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[GSPoint.m_Index].m_Left;
    }
    else
    {
      return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[GSPoint.m_Index].m_Right;
    }
  }
  assert ( m_ClassifierProjectedPoints[GSPoint.m_FrameNumber].m_Points.size() ==
    m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points.size() );
  std::vector < cv::Point2d > pointVector;
  for ( unsigned int i = 0 ; i < m_ClassifierProjectedPoints[GSPoint.m_FrameNumber].m_Points.size() ; i ++ )
  {
    if ( left )
    {
      pointVector.push_back ( m_ClassifierProjectedPoints[GSPoint.m_FrameNumber].m_Points[i].m_Left );
    }
    else
    {
      pointVector.push_back ( m_ClassifierProjectedPoints[GSPoint.m_FrameNumber].m_Points[i].m_Left );
    }
  }
  unsigned int myIndex;
  if ( ! boost::math::isinf(mitk::FindNearestPoint( GSPoint.m_Point , pointVector ,minRatio, &myIndex ).x))
  {
    if ( index != NULL ) 
    {
      *index = myIndex;
    }
    if ( left ) 
    {
      return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[myIndex].m_Left;
    }
    else
    {
      return m_ProjectedPoints[GSPoint.m_FrameNumber].m_Points[myIndex].m_Left;
    }
  }
  else
  {
    return cv::Point2d ( std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity() ) ;
  }
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetWorldPoints ( 
    std::vector < mitk::WorldPoint > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_WorldPoints.push_back(points[i]);
  }
  m_ProjectOK = false;
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetClassifierWorldPoints ( 
    std::vector < mitk::WorldPoint > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_ClassifierWorldPoints.push_back(points[i]);
  }
  m_ProjectOK = false;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetWorldPointsByTriangulation
    (std::vector< std::pair<cv::Point2d,cv::Point2d> > onScreenPointPairs,
     std::vector< unsigned int > framenumber  , mitk::VideoTrackerMatching::Pointer trackerMatcher, 
     std::vector<double> * perturbation)
{
  assert ( framenumber.size() == onScreenPointPairs.size() );

  if ( ! trackerMatcher->IsReady () ) 
  {
    MITK_ERROR << "Attempted to triangulate points without tracking matrices.";
    return;
  }
  
  cv::Mat * twoDPointsLeft = new  cv::Mat(onScreenPointPairs.size(),2,CV_64FC1);
  cv::Mat * twoDPointsRight =new  cv::Mat(onScreenPointPairs.size(),2,CV_64FC1);

  for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
  {
    twoDPointsLeft->at<double>( i, 0) = onScreenPointPairs[i].first.x;
    twoDPointsLeft->at<double> ( i , 1 ) = onScreenPointPairs[i].first.y;
    twoDPointsRight->at<double>( i , 0 ) = onScreenPointPairs[i].second.x;
    twoDPointsRight->at<double>( i , 1 ) = onScreenPointPairs[i].second.y;
  }
  cv::Mat leftScreenPoints = cv::Mat (onScreenPointPairs.size(),2,CV_64FC1);
  cv::Mat rightScreenPoints = cv::Mat (onScreenPointPairs.size(),2,CV_64FC1);

  bool cropUndistortedPointsToScreen = true;
  double cropValue = std::numeric_limits<double>::quiet_NaN();
  mitk::UndistortPoints(*twoDPointsLeft,
             *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,leftScreenPoints,
             cropUndistortedPointsToScreen , 
             0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);

  mitk::UndistortPoints(*twoDPointsRight,
             *m_RightIntrinsicMatrix,*m_RightDistortionVector,rightScreenPoints,
             cropUndistortedPointsToScreen , 
             0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
  
  cv::Mat leftCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat leftCameraRotationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat rightCameraTranslationVector = cv::Mat (3,1,CV_64FC1);
  cv::Mat rightCameraRotationVector = cv::Mat (3,1,CV_64FC1);

  for ( int i = 0 ; i < 3 ; i ++ )
  {
    leftCameraTranslationVector.at<double>(i,0) = 0.0;
    leftCameraRotationVector.at<double>(i,0) = 0.0;
  }
  rightCameraTranslationVector = *m_RightToLeftTranslationVector * -1;
  cv::Rodrigues ( m_RightToLeftRotationMatrix->inv(), rightCameraRotationVector  );

  CvMat leftScreenPointsMat = leftScreenPoints;
  CvMat rightScreenPointsMat= rightScreenPoints;
  CvMat leftCameraIntrinsicMat= *m_LeftIntrinsicMatrix;
  CvMat leftCameraRotationVectorMat= leftCameraRotationVector;
  CvMat leftCameraTranslationVectorMat= leftCameraTranslationVector;
  CvMat rightCameraIntrinsicMat = *m_RightIntrinsicMatrix;
  CvMat rightCameraRotationVectorMat = rightCameraRotationVector;
  CvMat rightCameraTranslationVectorMat= rightCameraTranslationVector;
  CvMat* leftCameraTriangulatedWorldPoints = cvCreateMat (onScreenPointPairs.size(),3,CV_64FC1);

  mitk::CStyleTriangulatePointPairsUsingSVD(
    leftScreenPointsMat,
    rightScreenPointsMat,
    leftCameraIntrinsicMat,
    leftCameraRotationVectorMat,
    leftCameraTranslationVectorMat,
    rightCameraIntrinsicMat,
    rightCameraRotationVectorMat,
    rightCameraTranslationVectorMat,
    *leftCameraTriangulatedWorldPoints);

  mitk::WorldPoint point;
  unsigned int wpSize=m_WorldPoints.size();
  for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
  {
    point = mitk::WorldPoint (
          cv::Point3d (
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,0),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,1),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,2) ),
          cv::Scalar(255,0,0)) ;
    long long timingError;
    point =  trackerMatcher->GetCameraTrackingMatrix(framenumber[i] , &timingError , m_TrackerIndex, perturbation, m_ReferenceIndex) * point;
    if ( abs(timingError) < m_AllowableTimingError )
    {
      m_WorldPoints.push_back ( point );
      MITK_INFO << framenumber[i] << " " << onScreenPointPairs[i].first << ","
        << onScreenPointPairs[i].second << " => " << point.m_Point << " => " << m_WorldPoints[i+wpSize].m_Point;
    }
    else
    {
      MITK_WARN << framenumber[i] << "Point rejected due to excessive timing error: " << timingError << " > " << m_AllowableTimingError;
    }


  }
  cvReleaseMat (&leftCameraTriangulatedWorldPoints);
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
  m_WorldPoints.clear();
  m_ProjectOK = false;
}

} // end namespace
