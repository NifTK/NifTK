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
#include <cv.h>
#include <highgui.h>

#include <boost/filesystem.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::ProjectPointsOnStereoVideo()
: m_Visualise(false)
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_Directory("")
, m_TrackerIndex(0)
, m_TrackerMatcher(NULL)
, m_DrawLines(false)
, m_InitOK(false)
, m_ProjectOK(false)
, m_DrawAxes(false)
, m_LeftIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, m_LeftDistortionVector (new cv::Mat(5,1,CV_32FC1))
, m_RightIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, m_RightDistortionVector (new cv::Mat(5,1,CV_32FC1))
, m_RightToLeftRotationMatrix (new cv::Mat(3,3,CV_32FC1))
, m_RightToLeftTranslationVector (new cv::Mat(3,1,CV_32FC1))
, m_LeftCameraToTracker (new cv::Mat(4,4,CV_32FC1))
, m_Capture(NULL)
, m_Writer(NULL)
{
}


//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::~ProjectPointsOnStereoVideo()
{

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
  if ( m_TrackerMatcher.IsNull() ) 
  {
    m_TrackerMatcher = mitk::VideoTrackerMatching::New();
  }
  if ( ! m_TrackerMatcher->IsReady() )
  {
    m_TrackerMatcher->Initialise(m_Directory);
  }
  if ( ! m_TrackerMatcher->IsReady() )
  {
    MITK_ERROR << "Failed to initialise tracker matcher";
    m_InitOK = false;
    return;
  }
  
  m_TrackerMatcher->SetCameraToTracker(*m_LeftCameraToTracker);
  if ( m_Visualise || m_SaveVideo ) 
  {
    if ( m_Capture == NULL ) 
    {
      std::vector <std::string> videoFiles = FindVideoData();
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
void ProjectPointsOnStereoVideo::Project()
{
  if ( ! m_InitOK )
  {
    MITK_WARN << "Called project before initialise.";
    return;
  }
    
  m_ProjectOK = false;
  m_ProjectedPoints.clear();
  m_PointsInLeftLensCS.clear();
  if ( m_WorldPoints.size() == 0 ) 
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
  while ( framenumber < m_TrackerMatcher->GetNumberOfFrames() && key != 'q')
  {
    //put the world points into the coordinates of the left hand camera.
    //worldtotracker * trackertocamera
    //in general the tracker matrices are trackertoworld
    cv::Mat WorldToLeftCamera = m_TrackerMatcher->GetCameraTrackingMatrix(framenumber, NULL, m_TrackerIndex).inv();
    
    std::vector < cv::Point3f > pointsInLeftLensCS = WorldToLeftCamera * m_WorldPoints; 
    m_PointsInLeftLensCS.push_back (pointsInLeftLensCS); 

    //project onto screen
    CvMat* outputLeftCameraWorldPointsIn3D = NULL;
    CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
    CvMat* output2DPointsLeft = NULL ;
    CvMat* output2DPointsRight = NULL;
    
    cv::Mat leftCameraWorldPoints = cv::Mat (pointsInLeftLensCS.size(),3,CV_32FC1);
    cv::Mat leftCameraWorldNormals = cv::Mat (pointsInLeftLensCS.size(),3,CV_32FC1);

    for ( unsigned int i = 0 ; i < pointsInLeftLensCS.size() ; i ++ ) 
    {
      leftCameraWorldPoints.at<float>(i,0) = pointsInLeftLensCS[i].x;
      leftCameraWorldPoints.at<float>(i,1) = pointsInLeftLensCS[i].y;
      leftCameraWorldPoints.at<float>(i,2) = pointsInLeftLensCS[i].z;
      leftCameraWorldNormals.at<float>(i,0) = 0.0;
      leftCameraWorldNormals.at<float>(i,1) = 0.0;
      leftCameraWorldNormals.at<float>(i,2) = -1.0;
    }
    cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_32FC1);
    leftCameraPositionToFocalPointUnitVector.at<float>(0,0)=0.0;
    leftCameraPositionToFocalPointUnitVector.at<float>(0,1)=0.0;
    leftCameraPositionToFocalPointUnitVector.at<float>(0,2)=1.0;
  
    mitk::ProjectVisible3DWorldPointsToStereo2D
      ( leftCameraWorldPoints,leftCameraWorldNormals,
        leftCameraPositionToFocalPointUnitVector,
        *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,
        *m_RightIntrinsicMatrix,*m_RightDistortionVector,
        *m_RightToLeftRotationMatrix,*m_RightToLeftTranslationVector,
        outputLeftCameraWorldPointsIn3D,
        outputLeftCameraWorldNormalsIn3D,
        output2DPointsLeft,
        output2DPointsRight);

    std::vector < std::pair < cv::Point2f , cv::Point2f > > screenPoints;
    
    for ( unsigned int i = 0 ; i < pointsInLeftLensCS.size() ; i ++ ) 
    {
      std::pair<cv::Point2f, cv::Point2f>  pointPair;
      pointPair.first = cv::Point2f(CV_MAT_ELEM(*output2DPointsLeft,float,i,0),CV_MAT_ELEM(*output2DPointsLeft,float,i,1));
      pointPair.second = cv::Point2f(CV_MAT_ELEM(*output2DPointsRight,float,i,0),CV_MAT_ELEM(*output2DPointsRight,float,i,1));
      screenPoints.push_back(pointPair);
    }
    m_ProjectedPoints.push_back(screenPoints);
    




    if ( m_Visualise || m_SaveVideo ) 
    {
      cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
      MITK_INFO << framenumber << " " << pointsInLeftLensCS[0] << " => " << screenPoints[0].first << screenPoints[0].second;
      if ( drawProjection )
      {
        if ( ! m_DrawLines ) 
        {
          if ( framenumber % 2 == 0 ) 
          {
            for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
            {
              cv::circle(videoImage, screenPoints[i].first,10, cvScalar(255,0,0), 3, 8, 0 );
            }
          }
          else
          {
            for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
            {
              cv::circle(videoImage, screenPoints[i].second,10, cvScalar(255,0,0), 3, 8, 0 );
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
              cv::line(videoImage, screenPoints[i].first,screenPoints[i+1].first, cvScalar(255,0,0) );
            }
            cv::line(videoImage, screenPoints[i].first,screenPoints[0].first, cvScalar(255,0,0) );
          }
          else
          {
            unsigned int i;
            for ( i = 0 ; i < screenPoints.size()-1 ; i ++ ) 
            {
              cv::line(videoImage, screenPoints[i].second,screenPoints[i+1].second, cvScalar(255,0,0) );
            } 
            cv::line(videoImage, screenPoints[i].second,screenPoints[0].second, cvScalar(255,0,0) );
          }
        }
        if ( m_DrawAxes && drawProjection )
        {
          if ( framenumber % 2 == 0 )
          {
            cv::line(videoImage,m_ScreenAxesPoints[0].first,m_ScreenAxesPoints[1].first,cvScalar(255,0,0));
            cv::line(videoImage,m_ScreenAxesPoints[0].first,m_ScreenAxesPoints[2].first,cvScalar(0,255,0));
            cv::line(videoImage,m_ScreenAxesPoints[0].first,m_ScreenAxesPoints[3].first,cvScalar(0,0,255));         
          }
          else
          {
            cv::line(videoImage,m_ScreenAxesPoints[0].second,m_ScreenAxesPoints[1].second,cvScalar(255,0,0));
            cv::line(videoImage,m_ScreenAxesPoints[0].second,m_ScreenAxesPoints[2].second,cvScalar(0,255,0));
            cv::line(videoImage,m_ScreenAxesPoints[0].second,m_ScreenAxesPoints[3].second,cvScalar(0,0,255));         
          }
        }
      }
      if ( m_Visualise ) 
      {
        IplImage image(videoImage);
        IplImage *smallimage = cvCreateImage (cvSize(960, 270), 8,3);
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
  m_ProjectOK = true;

}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetWorldPointsByTriangulation
    (std::vector< std::pair<cv::Point2f,cv::Point2f> > onScreenPointPairs,
     unsigned int framenumber)
{
  if ( ! m_TrackerMatcher->IsReady () ) 
  {
    MITK_ERROR << "Attempted to triangulate points without tracking matrices.";
    return;
  }
  
  cv::Mat * twoDPointsLeft = new  cv::Mat(onScreenPointPairs.size(),2,CV_32FC1);
  cv::Mat * twoDPointsRight =new  cv::Mat(onScreenPointPairs.size(),2,CV_32FC1);

  for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
  {
    twoDPointsLeft->at<float>( i, 0) = onScreenPointPairs[i].first.x;
    twoDPointsLeft->at<float> ( i , 1 ) = onScreenPointPairs[i].first.y;
    twoDPointsRight->at<float>( i , 0 ) = onScreenPointPairs[i].second.x;
    twoDPointsRight->at<float>( i , 1 ) = onScreenPointPairs[i].second.y;
  }
  cv::Mat leftScreenPoints = cv::Mat (onScreenPointPairs.size(),2,CV_32FC1);
  cv::Mat rightScreenPoints = cv::Mat (onScreenPointPairs.size(),2,CV_32FC1);

  mitk::UndistortPoints(*twoDPointsLeft,
             *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,leftScreenPoints);

  mitk::UndistortPoints(*twoDPointsRight,
             *m_RightIntrinsicMatrix,*m_RightDistortionVector,rightScreenPoints);
  
  cv::Mat leftCameraTranslationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat leftCameraRotationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat rightCameraTranslationVector = cv::Mat (3,1,CV_32FC1);
  cv::Mat rightCameraRotationVector = cv::Mat (3,1,CV_32FC1);

  for ( int i = 0 ; i < 3 ; i ++ )
  {
    leftCameraTranslationVector.at<float>(i,0) = 0.0;
    leftCameraRotationVector.at<float>(i,0) = 0.0;
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
  CvMat* leftCameraTriangulatedWorldPoints = cvCreateMat (onScreenPointPairs.size(),3,CV_32FC1);

  mitk::TriangulatePointPairs(
    leftScreenPointsMat,
    rightScreenPointsMat,
    leftCameraIntrinsicMat,
    leftCameraRotationVectorMat,
    leftCameraTranslationVectorMat,
    rightCameraIntrinsicMat,
    rightCameraRotationVectorMat,
    rightCameraTranslationVectorMat,
    *leftCameraTriangulatedWorldPoints);

  std::vector <cv::Point3f> points;
  for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
  {
    points.push_back(cv::Point3f (
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,0),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,1),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,2) ) ) ;
  }

  m_WorldPoints = m_TrackerMatcher->GetCameraTrackingMatrix(framenumber , NULL , m_TrackerIndex) * points;

  for ( unsigned int i = 0 ; i < onScreenPointPairs.size(); i ++ ) 
  {
    MITK_INFO << framenumber << " " << onScreenPointPairs[i].first << ","
      << onScreenPointPairs[i].second << " => " << points[i] << " => " << m_WorldPoints[i];
  }

}

//-----------------------------------------------------------------------------
std::vector<std::string> ProjectPointsOnStereoVideo::FindVideoData()
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
                                          
void ProjectPointsOnStereoVideo::SetFlipMatrices(bool state)
{
  if ( m_TrackerMatcher.IsNull() ) 
  {
    MITK_ERROR << "Tried to set flip matrices before initialisation";
    return;
  }
  m_TrackerMatcher->SetFlipMatrices(state);
}

std::vector < std::vector <cv::Point3f> > ProjectPointsOnStereoVideo::GetPointsInLeftLensCS()
{
  return m_PointsInLeftLensCS;
}

std::vector < std::vector <std::pair <cv::Point2f, cv::Point2f> > > ProjectPointsOnStereoVideo::GetProjectedPoints()
{
  return m_ProjectedPoints;
}
void ProjectPointsOnStereoVideo::ProjectAxes()
{
  cv::Mat leftCameraAxesPoints = cv::Mat (4,3,CV_32FC1);
  cv::Mat leftCameraAxesNormals = cv::Mat (4,3,CV_32FC1);
  for ( int i = 0 ; i < 4 ; i ++ ) 
  {
    int j;
    for ( j = 0 ; j < 2 ; j ++ ) 
    {
      leftCameraAxesPoints.at<float>(i,j) = 0.0;
      leftCameraAxesNormals.at<float>(i,j) = 0.0;
    }
      leftCameraAxesPoints.at<float>(i,j) = 100.0;
      leftCameraAxesNormals.at<float>(i,j) = -1.0;
  }
  leftCameraAxesPoints.at<float>(1,0) = 100;
  leftCameraAxesPoints.at<float>(2,1) = 100;
  leftCameraAxesPoints.at<float>(3,2) = 200;
      
  CvMat* outputLeftCameraAxesPointsIn3D = NULL;
  CvMat* outputLeftCameraAxesNormalsIn3D = NULL ;
  CvMat* output2DAxesPointsLeft = NULL ;
  CvMat* output2DAxesPointsRight = NULL;
  
  cv::Mat CameraUnitVector = cv::Mat(1,3,CV_32FC1);
  CameraUnitVector.at<float>(0,0)=0;
  CameraUnitVector.at<float>(0,1)=0;
  CameraUnitVector.at<float>(0,2)=1.0;
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
    std::pair<cv::Point2f, cv::Point2f>  pointPair;
    pointPair.first = cv::Point2f(CV_MAT_ELEM(*output2DAxesPointsLeft,float,i,0),CV_MAT_ELEM(*output2DAxesPointsLeft,float,i,1));
    pointPair.second = cv::Point2f(CV_MAT_ELEM(*output2DAxesPointsRight,float,i,0),CV_MAT_ELEM(*output2DAxesPointsRight,float,i,1));
    MITK_INFO << "Left" << pointPair.first << "Right" << pointPair.second;

    m_ScreenAxesPoints.push_back(pointPair);
  }
}
//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetVideoLagMilliseconds ( unsigned long long VideoLag, bool VideoLeadsTracking)
{
  if ( m_TrackerMatcher.IsNull()  || (! m_TrackerMatcher->IsReady()) )
  {
    MITK_ERROR << "Need to initialise before setting video lag";
    return;
  } 
  m_TrackerMatcher->SetVideoLagMilliseconds (VideoLag, VideoLeadsTracking);
}

} // end namespace
