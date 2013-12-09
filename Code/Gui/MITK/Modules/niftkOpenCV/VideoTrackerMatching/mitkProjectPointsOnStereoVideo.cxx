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
, m_TrackerIndex(0)
, m_ReferenceIndex(-1)
, m_DrawLines(false)
, m_InitOK(false)
, m_ProjectOK(false)
, m_DrawAxes(false)
, m_LeftIntrinsicMatrix (new cv::Mat(3,3,CV_64FC1))
, m_LeftDistortionVector (new cv::Mat(1,4,CV_64FC1))
, m_RightIntrinsicMatrix (new cv::Mat(3,3,CV_64FC1))
, m_RightDistortionVector (new cv::Mat(1,4,CV_64FC1))
, m_RightToLeftRotationMatrix (new cv::Mat(3,3,CV_64FC1))
, m_RightToLeftTranslationVector (new cv::Mat(3,1,CV_64FC1))
, m_LeftCameraToTracker (new cv::Mat(4,4,CV_64FC1))
, m_Capture(NULL)
, m_LeftWriter(NULL)
, m_RightWriter(NULL)
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
  if ( savevideo )
  {
    cv::Size S = cv::Size((int) 960, (int) 540 );
    m_LeftWriter =cvCreateVideoWriter(std::string(prefix + "leftchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),15,S, true);
    m_RightWriter =cvCreateVideoWriter(std::string(prefix + "rightchannel.avi").c_str(), CV_FOURCC('D','I','V','X'),15,S, true);
  }
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
  IplImage *smallimage = cvCreateImage (cvSize(960, 270), 8,3);
  IplImage *smallcorrectedimage = cvCreateImage (cvSize(960, 540), 8,3);
  while ( framenumber < trackerMatcher->GetNumberOfFrames() && key != 'q')
  {
    //put the world points into the coordinates of the left hand camera.
    //worldtotracker * trackertocamera
    //in general the tracker matrices are trackertoworld
    cv::Mat WorldToLeftCamera = trackerMatcher->GetCameraTrackingMatrix(framenumber, NULL, m_TrackerIndex, perturbation, m_ReferenceIndex).inv();
    
    m_WorldToLeftCameraMatrices.push_back(WorldToLeftCamera);
    std::vector < std::pair < cv::Point3d , cv::Scalar > > pointsInLeftLensCS = WorldToLeftCamera * m_WorldPoints; 
    m_PointsInLeftLensCS.push_back (pointsInLeftLensCS); 

    //project onto screen
    CvMat* outputLeftCameraWorldPointsIn3D = NULL;
    CvMat* outputLeftCameraWorldNormalsIn3D = NULL ;
    CvMat* output2DPointsLeft = NULL ;
    CvMat* output2DPointsRight = NULL;
    
    cv::Mat leftCameraWorldPoints = cv::Mat (pointsInLeftLensCS.size(),3,CV_64FC1);
    cv::Mat leftCameraWorldNormals = cv::Mat (pointsInLeftLensCS.size(),3,CV_64FC1);

    for ( unsigned int i = 0 ; i < pointsInLeftLensCS.size() ; i ++ ) 
    {
      leftCameraWorldPoints.at<double>(i,0) = pointsInLeftLensCS[i].first.x;
      leftCameraWorldPoints.at<double>(i,1) = pointsInLeftLensCS[i].first.y;
      leftCameraWorldPoints.at<double>(i,2) = pointsInLeftLensCS[i].first.z;
      leftCameraWorldNormals.at<double>(i,0) = 0.0;
      leftCameraWorldNormals.at<double>(i,1) = 0.0;
      leftCameraWorldNormals.at<double>(i,2) = -1.0;
    }
    cv::Mat leftCameraPositionToFocalPointUnitVector = cv::Mat(1,3,CV_64FC1);
    leftCameraPositionToFocalPointUnitVector.at<double>(0,0)=0.0;
    leftCameraPositionToFocalPointUnitVector.at<double>(0,1)=0.0;
    leftCameraPositionToFocalPointUnitVector.at<double>(0,2)=1.0;
  
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

    std::vector < std::pair < cv::Point2d , cv::Point2d > > screenPoints;
    
    for ( unsigned int i = 0 ; i < pointsInLeftLensCS.size() ; i ++ ) 
    {
      std::pair<cv::Point2d, cv::Point2d>  pointPair;
      pointPair.first = cv::Point2d(CV_MAT_ELEM(*output2DPointsLeft,double,i,0),CV_MAT_ELEM(*output2DPointsLeft,double,i,1));
      pointPair.second = cv::Point2d(CV_MAT_ELEM(*output2DPointsRight,double,i,0),CV_MAT_ELEM(*output2DPointsRight,double,i,1));
      screenPoints.push_back(pointPair);
    }
    m_ProjectedPoints.push_back(screenPoints);
    
    //de-allocate the matrices    
    cvReleaseMat(&outputLeftCameraWorldPointsIn3D);
    cvReleaseMat(&outputLeftCameraWorldNormalsIn3D);
    cvReleaseMat(&output2DPointsLeft);
    cvReleaseMat(&output2DPointsRight);

    if ( m_Visualise || m_SaveVideo ) 
    {
      cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
      MITK_INFO << framenumber << " " << m_WorldPoints[0].first << " " << pointsInLeftLensCS[0].first << " => " << screenPoints[0].first << screenPoints[0].second;
      if ( drawProjection )
      {
        if ( ! m_DrawLines ) 
        {
          if ( framenumber % 2 == 0 ) 
          {
            for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
            {
              cv::circle(videoImage, screenPoints[i].first,10, pointsInLeftLensCS[i].second, 3, 8, 0 );
            }
          }
          else
          {
            for ( unsigned int i = 0 ; i < screenPoints.size() ; i ++ ) 
            {
              cv::circle(videoImage, screenPoints[i].second,10, pointsInLeftLensCS[i].second, 3, 8, 0 );
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
              cv::line(videoImage, screenPoints[i].first,screenPoints[i+1].first, pointsInLeftLensCS[i].second );
            }
            cv::line(videoImage, screenPoints[i].first,screenPoints[0].first, pointsInLeftLensCS[i].second );
          }
          else
          {
            unsigned int i;
            for ( i = 0 ; i < screenPoints.size()-1 ; i ++ ) 
            {
              cv::line(videoImage, screenPoints[i].second,screenPoints[i+1].second, pointsInLeftLensCS[i].second );
            } 
            cv::line(videoImage, screenPoints[i].second,screenPoints[0].second, pointsInLeftLensCS[i].second );
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
    std::vector < std::pair < unsigned int , cv::Point2d > > points )
{
   m_LeftGoldStandardPoints = points;
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetRightGoldStandardPoints (
    std::vector < std::pair < unsigned int , cv::Point2d > > points )
{
   m_RightGoldStandardPoints = points;
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
    m_LeftProjectionErrors.push_back(this->CalculateProjectionError( m_LeftGoldStandardPoints[i], left) );
    m_LeftReProjectionErrors.push_back(this->CalculateReProjectionError ( m_LeftGoldStandardPoints[i] , left ));
  }
  for ( unsigned int i = 0 ; i < m_RightGoldStandardPoints.size() ; i ++ ) 
  {
    bool left = false;
    m_RightProjectionErrors.push_back(this->CalculateProjectionError( m_RightGoldStandardPoints[i], left) );
    m_RightReProjectionErrors.push_back(this->CalculateReProjectionError ( m_RightGoldStandardPoints[i], left ));
  }

  std::ofstream lpout (std::string (outPrefix + "_leftProjection.errors").c_str());
  lpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_LeftProjectionErrors.size() ; i ++ )
  {
    lpout << m_LeftProjectionErrors[i] ;
  }
  lpout.close();

  std::ofstream rpout (std::string (outPrefix + "_rightProjection.errors").c_str());
  rpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_RightProjectionErrors.size() ; i ++ )
  {
    rpout << m_RightProjectionErrors[i] ;
  }
  rpout.close();

  std::ofstream lrpout (std::string (outPrefix + "_leftReProjection.errors").c_str());
  lrpout << "#xmm ymm zmm" << std::endl;
  for ( unsigned int i = 0 ; i < m_LeftReProjectionErrors.size() ; i ++ )
  {
    lrpout << m_LeftReProjectionErrors[i] ;
  }
  lrpout.close();

  std::ofstream rrpout (std::string (outPrefix + "_rightReProjection.errors").c_str());
  rrpout << "#xpixels ypixels" << std::endl;
  for ( unsigned int i = 0 ; i < m_RightReProjectionErrors.size() ; i ++ )
  {
    rrpout << m_RightReProjectionErrors[i] ;
  }
  rrpout.close();

}

//-----------------------------------------------------------------------------
cv::Point3d ProjectPointsOnStereoVideo::CalculateReProjectionError ( std::pair < unsigned int, cv::Point2d > GSPoint, bool left )
{
  unsigned int* index = new unsigned int;
  cv::Point2d matchingPoint = FindNearestScreenPoint ( GSPoint, left, index ) ;
  cv::Point3d matchingPointInLensCS = m_PointsInLeftLensCS[GSPoint.first][*index].first;

  if ( ! left )
  {
    cv::Mat m1 = cvCreateMat(3,1,CV_64FC1);
    m1.at<double>(0,0) = matchingPointInLensCS.x;
    m1.at<double>(1,0) = matchingPointInLensCS.y;
    m1.at<double>(2,0) = matchingPointInLensCS.z;

    m1 = m1 * m_RightToLeftRotationMatrix->inv() - *m_RightToLeftTranslationVector;
    
    matchingPointInLensCS.x = m1.at<double>(0,0);
    matchingPointInLensCS.y = m1.at<double>(1,0);
    matchingPointInLensCS.z = m1.at<double>(2,0);
  }
  
  cv::Point3d reProjectionGS;
  if ( left ) 
  {
    cv::Point2d undistortedPoint;
    mitk::UndistortPoint (GSPoint.second, *m_LeftIntrinsicMatrix, 
        *m_LeftDistortionVector, undistortedPoint);
    reProjectionGS = mitk::ReProjectPoint (undistortedPoint , *m_LeftIntrinsicMatrix);
  }
  else
  {
    cv::Point2d undistortedPoint;
    mitk::UndistortPoint (GSPoint.second, *m_RightIntrinsicMatrix, 
        *m_RightDistortionVector, undistortedPoint);
    reProjectionGS = mitk::ReProjectPoint (undistortedPoint , *m_RightIntrinsicMatrix);
  }
  
  reProjectionGS.x *= matchingPointInLensCS.z;
  reProjectionGS.y *= matchingPointInLensCS.z;
  reProjectionGS.z *= matchingPointInLensCS.z;

  delete index;
  return matchingPointInLensCS - reProjectionGS;
}

//-----------------------------------------------------------------------------
cv::Point2d ProjectPointsOnStereoVideo::CalculateProjectionError ( std::pair < unsigned int, cv::Point2d > GSPoint, bool left )
{
  cv::Point2d matchingPoint = FindNearestScreenPoint ( GSPoint, left ) ;
  
  return matchingPoint - GSPoint.second;

}

//-----------------------------------------------------------------------------
cv::Point2d ProjectPointsOnStereoVideo::FindNearestScreenPoint ( std::pair < unsigned int, cv::Point2d> GSPoint, bool left , unsigned int* index)
{
  std::vector < cv::Point2d > pointVector;
  for ( unsigned int i = 0 ; i < m_ProjectedPoints[GSPoint.first].size() ; i ++ )
  {
    if ( left )
    {
      pointVector.push_back ( m_ProjectedPoints[GSPoint.first][i].first );
    }
    else
    {
      pointVector.push_back ( m_ProjectedPoints[GSPoint.first][i].second );
    }
  }
  return mitk::FindNearestPoint( GSPoint.second , pointVector , index );
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetWorldPoints ( 
    std::vector < std::pair < cv::Point3d , cv::Scalar > > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_WorldPoints.push_back(points[i]);
  }
}

//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::SetWorldPoints ( 
    std::vector < cv::Point3d > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    std::pair < cv::Point3d, cv::Scalar > point = 
      std::pair < cv::Point3d , cv::Scalar > ( points[i], cv::Scalar (255,0,0) );
    m_WorldPoints.push_back(point);
  }
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

  mitk::UndistortPoints(*twoDPointsLeft,
             *m_LeftIntrinsicMatrix,*m_LeftDistortionVector,leftScreenPoints);

  mitk::UndistortPoints(*twoDPointsRight,
             *m_RightIntrinsicMatrix,*m_RightDistortionVector,rightScreenPoints);
  
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

  std::pair  <cv::Point3d, cv::Scalar > point;
  unsigned int wpSize=m_WorldPoints.size();
  for ( unsigned int i = 0 ; i < onScreenPointPairs.size() ; i ++ ) 
  {
    point = std::pair < cv::Point3d , cv::Scalar > (
          cv::Point3d (
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,0),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,1),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,double,i,2) ),
          cv::Scalar(255,0,0)) ;
    m_WorldPoints.push_back ( trackerMatcher->GetCameraTrackingMatrix(framenumber[i] , NULL , m_TrackerIndex, perturbation, m_ReferenceIndex) * point );
    MITK_INFO << framenumber[i] << " " << onScreenPointPairs[i].first << ","
      << onScreenPointPairs[i].second << " => " << point.first << " => " << m_WorldPoints[i-wpSize].first;
  }

}

std::vector < std::vector <cv::Point3d> > ProjectPointsOnStereoVideo::GetPointsInLeftLensCS()
{
  std::vector < std::vector < cv::Point3d > > returnPoints;
  for ( unsigned int i = 0 ; i < m_PointsInLeftLensCS.size() ; i ++ ) 
  {
    std::vector < cv::Point3d > thesePoints;
    for ( unsigned int j = 0 ; j < m_PointsInLeftLensCS[i].size() ; j ++ ) 
    {
      thesePoints.push_back ( m_PointsInLeftLensCS[i][j].first );
    }
    returnPoints.push_back(thesePoints);
  }
  return returnPoints;
}

std::vector < std::vector <std::pair <cv::Point2d, cv::Point2d> > > ProjectPointsOnStereoVideo::GetProjectedPoints()
{
  return m_ProjectedPoints;
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
    std::pair<cv::Point2d, cv::Point2d>  pointPair;
    pointPair.first = cv::Point2d(CV_MAT_ELEM(*output2DAxesPointsLeft,double,i,0),CV_MAT_ELEM(*output2DAxesPointsLeft,double,i,1));
    pointPair.second = cv::Point2d(CV_MAT_ELEM(*output2DAxesPointsRight,double,i,0),CV_MAT_ELEM(*output2DAxesPointsRight,double,i,1));
    MITK_INFO << "Left" << pointPair.first << "Right" << pointPair.second;

    m_ScreenAxesPoints.push_back(pointPair);
  }
}

} // end namespace
