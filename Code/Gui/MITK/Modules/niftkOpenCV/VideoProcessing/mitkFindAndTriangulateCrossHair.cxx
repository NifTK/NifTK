/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkFindAndTriangulateCrossHair.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <cv.h>
#include <highgui.h>

#include <boost/filesystem.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
FindAndTriangulateCrossHair::FindAndTriangulateCrossHair()
: m_Visualise(false)
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_Directory("")
, m_TrackerIndex(0)
, m_TrackerMatcher(NULL)
, m_InitOK(false)
, m_TriangulateOK(false)
, m_CalibrateOK(false)
, m_TrackingOK(false)
, m_LeftIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, m_LeftDistortionVector (new cv::Mat(5,1,CV_32FC1))
, m_RightIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, m_RightDistortionVector (new cv::Mat(5,1,CV_32FC1))
, m_RightToLeftRotationMatrix (new cv::Mat(3,3,CV_32FC1))
, m_RightToLeftTranslationVector (new cv::Mat(3,1,CV_32FC1))
, m_LeftCameraToTracker (new cv::Mat(4,4,CV_32FC1))
, m_Capture(NULL)
, m_Writer(NULL)
, m_BlurKernel (cv::Size (3,3))
, m_HoughRho (1.0)
, m_HoughTheta(CV_PI/(180))
, m_HoughThreshold(50)
, m_HoughLineLength(130)
, m_HoughLineGap(20)
{
}


//-----------------------------------------------------------------------------
FindAndTriangulateCrossHair::~FindAndTriangulateCrossHair()
{

}

//-----------------------------------------------------------------------------
void FindAndTriangulateCrossHair::Initialise(std::string directory, 
    std::string calibrationParameterDirectory)
{
  m_InitOK = false;
  m_Directory = directory;

  try
  {
    m_CalibrateOK = true;
    mitk::LoadStereoCameraParametersFromDirectory
      ( calibrationParameterDirectory,
      m_LeftIntrinsicMatrix,m_LeftDistortionVector,m_RightIntrinsicMatrix,
      m_RightDistortionVector,m_RightToLeftRotationMatrix,
      m_RightToLeftTranslationVector,m_LeftCameraToTracker);
  }
  catch ( int e )
  {
    MITK_WARN << "Failed to load camera parameters";
    m_CalibrateOK = false;
  }
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
    MITK_WARN << "Failed to initialise tracker matcher";
    m_TrackingOK = false;
  }
  else 
  {
    m_TrackingOK = true;
  }

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
void FindAndTriangulateCrossHair::SetVisualise ( bool visualise )
{
  m_Visualise = visualise;
}
//-----------------------------------------------------------------------------
void FindAndTriangulateCrossHair::SetSaveVideo ( bool savevideo )
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
void FindAndTriangulateCrossHair::Triangulate()
{
  if ( ! m_InitOK )
  {
    MITK_WARN << "Called triangulate before initialise.";
    return;
  }
    
  m_TriangulateOK = false;
  m_WorldPoints.clear();
  m_PointsInLeftLensCS.clear();
  m_ScreenPoints.clear();

  if ( m_Visualise ) 
  {
    cvNamedWindow ("Left Channel", CV_WINDOW_AUTOSIZE);
    cvNamedWindow ("Right Channel", CV_WINDOW_AUTOSIZE);
    cvNamedWindow ("Processed Left", CV_WINDOW_AUTOSIZE);
  }
  int framenumber = 0 ;
  int key = 0;

  cv::Mat leftFrame;
  cv::Mat rightFrame;
  cv::Mat leftCanny;
  cv::Mat rightCanny;
  cv::Mat leftHough;
  cv::Mat rightHough;
  m_ScreenPoints.clear();
  while ( framenumber < m_TrackerMatcher->GetNumberOfFrames() && key != 'q')
  {

    cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
    leftFrame = videoImage.clone();
    videoImage = cvQueryFrame ( m_Capture ) ;
    rightFrame = videoImage.clone();
    
    cv::cvtColor( leftFrame, leftHough, CV_BGR2GRAY );
    cv::cvtColor( rightFrame, rightHough, CV_BGR2GRAY );
    int lowThreshold=20;
    int highThreshold = 70;
    int kernel = 3;
    cv::Canny(leftHough,leftCanny, lowThreshold,highThreshold,kernel);
    cv::Canny(rightHough,rightCanny, lowThreshold,highThreshold,kernel);
    cv::vector<cv::Vec4i> linesleft;
    cv::vector<cv::Vec4i> linesright;
    cv::HoughLinesP (leftCanny, linesleft,m_HoughRho,m_HoughTheta, m_HoughThreshold, m_HoughLineLength , m_HoughLineGap);  
    cv::HoughLinesP (rightCanny, linesright,m_HoughRho,m_HoughTheta, m_HoughThreshold, m_HoughLineLength , m_HoughLineGap);  
    std::pair <cv::Point2f, cv::Point2f> screenPoints;
    screenPoints.first = cv::Point2f(-100.0, -100.0);
    screenPoints.second = cv::Point2f(-100.0, -100.0);
    for ( int i = 0 ; i < linesleft.size() ; i ++ )
    {
      cv::line(leftFrame,cvPoint(linesleft[i][0],linesleft[i][1]),
          cvPoint(linesleft[i][2],linesleft[i][3]),cvScalar(255,0,0));
    }
    for ( int i = 0 ; i < linesright.size() ; i ++ )
    {
      cv::line(rightFrame,cvPoint(linesright[i][0],linesright[i][1]),
          cvPoint(linesright[i][2],linesright[i][3]),cvScalar(255,0,0));
    }
    std::vector <cv::Point2f> leftIntersectionPoints = mitk::FindIntersects (linesleft, true, true);
    std::vector <cv::Point2f> rightIntersectionPoints = mitk::FindIntersects (linesright, true, true);
    screenPoints.first = mitk::GetCentroid(leftIntersectionPoints,true);
    screenPoints.second = mitk::GetCentroid(rightIntersectionPoints,true);
    cv::circle(leftFrame , screenPoints.first,10, cvScalar(0,0,255),2,8,0);
    cv::circle(rightFrame , screenPoints.second,10, cvScalar(0,255,0),2,8,0);
    m_ScreenPoints.push_back(screenPoints);
    if ( m_Visualise ) 
    {
      IplImage *smallleft = cvCreateImage (cvSize(960, 270), 8,3);
      cvResize (&(IplImage(leftFrame)), smallleft,CV_INTER_LINEAR);
      IplImage *smallright = cvCreateImage (cvSize(960, 270), 8,3);
      cvResize (&(IplImage(rightFrame)), smallright,CV_INTER_LINEAR);
      IplImage *smallprocessed = cvCreateImage (cvSize(960, 270), 8,1);
      cvResize (&(IplImage(leftCanny)), smallprocessed , CV_INTER_LINEAR);
      cvShowImage("Left Channel" , smallleft);
      cvShowImage("Right Channel" , smallright);
      cvShowImage("Processed Left", smallprocessed);
      key = cvWaitKey (20);
      if ( key == 's' )
      {
        m_Visualise = false;
      }
    }

    framenumber ++;
    framenumber ++;
  }
  if ( m_ScreenPoints.size() !=  m_TrackerMatcher->GetNumberOfFrames()/2 )
  {
    MITK_ERROR << "Got the wrong number of screen point pairs " << m_ScreenPoints.size() 
      << " != " << m_TrackerMatcher->GetNumberOfFrames()/2;
    m_TriangulateOK = false;
  }
  else
  {
    m_TriangulateOK = true;
  }
}
//-----------------------------------------------------------------------------
void FindAndTriangulateCrossHair::TriangulatePoints()
{
  cv::Mat * twoDPointsLeft = new cv::Mat(m_ScreenPoints.size(),2,CV_32FC1);
  cv::Mat * twoDPointsRight = new cv::Mat(m_ScreenPoints.size(),2,CV_32FC1);

  for ( unsigned int i = 0 ; i < m_ScreenPoints.size() ; i ++ ) 
  {
    twoDPointsLeft->at<float>( i, 0) = m_ScreenPoints[i].first.x;
    twoDPointsLeft->at<float> ( i , 1 ) = m_ScreenPoints[i].first.y;
    twoDPointsRight->at<float>( i , 0 ) = m_ScreenPoints[i].second.x;
    twoDPointsRight->at<float>( i , 1 ) = m_ScreenPoints[i].second.y;
  }
  cv::Mat leftScreenPoints = cv::Mat (m_ScreenPoints.size(),2,CV_32FC1);
  cv::Mat rightScreenPoints = cv::Mat (m_ScreenPoints.size(),2,CV_32FC1);

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
  CvMat* leftCameraTriangulatedWorldPoints = cvCreateMat (m_ScreenPoints.size(),3,CV_32FC1);

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

  for ( unsigned int i = 0 ; i < m_ScreenPoints.size() ; i ++ ) 
  {
    m_PointsInLeftLensCS.push_back(cv::Point3f (
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,0),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,1),
        CV_MAT_ELEM(*leftCameraTriangulatedWorldPoints,float,i,2) ) ) ;
  }
  
  
  //m_WorldPoints = m_TrackerMatcher->GetCameraMatrix(framenumber, NULL , m_TrackerIndex) * points;

}

//-----------------------------------------------------------------------------
std::vector<std::string> FindAndTriangulateCrossHair::FindVideoData()
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
                                          
void FindAndTriangulateCrossHair::SetFlipMatrices(bool state)
{
  if ( m_TrackerMatcher.IsNull() ) 
  {
    MITK_ERROR << "Tried to set flip matrices before initialisation";
    return;
  }
  m_TrackerMatcher->SetFlipMatrices(state);
}

} // end namespace
