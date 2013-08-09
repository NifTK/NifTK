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
  }
  int framenumber = 0 ;
  int key = 0;
  while ( framenumber < m_TrackerMatcher->GetNumberOfFrames() && key != 'q')
  {
    cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
        
    std::pair <cv::Point2f, cv::Point2f> screenPoints;

    if ( framenumber % 2 == 0 ) 
    {
      cv::circle(videoImage, screenPoints.first,10, cvScalar(255,0,0), 3, 8, 0 );
    }
    else
    {
      cv::circle(videoImage, screenPoints.second,10, cvScalar(255,0,0), 3, 8, 0 );
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
    }
    framenumber ++;
  }
  m_TriangulateOK = true;
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
