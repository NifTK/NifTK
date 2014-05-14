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
, m_VideoWidth(1920)
, m_VideoHeight(540)
, m_Capture(NULL)
, m_AllowablePointMatchingRatio (1.0) 
, m_AllowableTimingError (20e6) // 20 milliseconds 
, m_StartFrame(0)
, m_EndFrame(0)
, m_ProjectorScreenBuffer(0.0)
, m_ClassifierScreenBuffer(100.0)
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
  ProjectAxes(); 
  
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

      cv::Mat videoImage = cvQueryFrame ( m_Capture ) ;
      MITK_INFO << framenumber << " " << timingError;
      
      IplImage image(videoImage);
      if ( framenumber %2 == 0 ) 
      {
        cvShowImage("Left Channel" , &image);
      }
      else
      {
        cvShowImage("Right Channel" , &image);
      }
      key = cvWaitKey (20);

      std::vector <cv::Point2d> pickedPoints;
      while ( key != 'n' )
      {
        cvSetMouseCallback("Left Channel",CallBackFunc, &pickedPoints);
        key = cvWaitKey(20);
        if ( pickedPoints.size() > 0 )
        {
          for ( int i = 0 ; i < pickedPoints.size() ; i ++ ) 
          {
            cv::circle(videoImage, pickedPoints[i],10,cv::Scalar(255,255,255),8,3);
          }
            
          IplImage image(videoImage);
          cvShowImage("Left Channel" , &image);
        }
      }
      std::string outPrefix = "outItGoes";
      std::ofstream pointOut (std::string (outPrefix + "_frame000.points").c_str());
      pointOut << pickedPoints;
      pointOut.close();
      exit(1);
    }
    framenumber ++;
  }
  m_ProjectOK = true;
}
//-----------------------------------------------------------------------------
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  std::vector<cv::Point2d>* out = static_cast<std::vector<cv::Point2d>*>(userdata);
  if  ( event == cv::EVENT_LBUTTONDOWN )
  {
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    out->push_back (cv::Point2d(x,y));
  }
  else if  ( event == cv::EVENT_RBUTTONDOWN )
  {
    std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  }
  else if  ( event == cv::EVENT_MBUTTONDOWN )
  {
    std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  }
  else if ( event == cv::EVENT_MOUSEMOVE )
  {
    std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
  }
}
//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetLeftGoldStandardPoints (
    std::vector < std::pair < unsigned int , cv::Point2d > > points )
{
   m_LeftGoldStandardPoints = points;
}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetRightGoldStandardPoints (
    std::vector < std::pair < unsigned int , cv::Point2d > > points )
{
   m_RightGoldStandardPoints = points;
}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::CalculateTriangulationErrors (std::string outPrefix, 
    mitk::VideoTrackerMatching::Pointer trackerMatcher)
{
  if ( ! m_ProjectOK ) 
  {
    MITK_ERROR << "Attempted to run CalculateProjectionErrors, before running project(), no result.";
    return;
  }

  std::sort ( m_LeftGoldStandardPoints.begin(), m_LeftGoldStandardPoints.end(), mitk::CompareGSPointPair);
  std::sort ( m_RightGoldStandardPoints.begin(), m_RightGoldStandardPoints.end() , mitk::CompareGSPointPair );
  
  unsigned int leftGSIndex = 0;
  unsigned int rightGSIndex = 0;

  std::vector < std::vector < cv::Point3d > > classifiedPoints;
  for ( unsigned int i = 0 ; i < m_PointsInLeftLensCS[0].second.size() ; i ++ )
  {
    std::vector < cv::Point3d > pointvector;
    classifiedPoints.push_back(pointvector);
  }

  while ( leftGSIndex < m_LeftGoldStandardPoints.size() && rightGSIndex < m_RightGoldStandardPoints.size() )
  {
    unsigned int frameNumber = m_LeftGoldStandardPoints[leftGSIndex].first;
    std::vector < cv::Point2d > leftPoints;
    std::vector < cv::Point2d > rightPoints;
    std::vector < std::pair < unsigned int , std::pair < cv::Point2d , cv::Point2d > > > matchedPairs;
    
    while ( m_LeftGoldStandardPoints[leftGSIndex].first == frameNumber && leftGSIndex < m_LeftGoldStandardPoints.size() ) 
    {
      leftPoints.push_back ( m_LeftGoldStandardPoints[leftGSIndex].second );
      leftGSIndex ++;
    }
    while ( m_RightGoldStandardPoints[rightGSIndex].first < frameNumber  && rightGSIndex < m_RightGoldStandardPoints.size() )  
    {
      rightGSIndex ++;
    }

    while ( m_RightGoldStandardPoints[rightGSIndex].first == frameNumber  && rightGSIndex < m_RightGoldStandardPoints.size() )  
    {
      rightPoints.push_back ( m_RightGoldStandardPoints[rightGSIndex].second );
      rightGSIndex ++;
    }
//check timing error here
    if ( abs (m_PointsInLeftLensCS[frameNumber].first) < m_AllowableTimingError )
    {
      for ( unsigned int i = 0 ; i < leftPoints.size() ; i ++ ) 
      {
        unsigned int index;
        double minRatio;
        bool left = true;
        this->FindNearestScreenPoint ( std::pair < unsigned int, cv::Point2d >
            ( frameNumber, leftPoints[i] ) , left, &minRatio, &index );
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
            this->FindNearestScreenPoint ( std::pair < unsigned int, cv::Point2d >
              ( frameNumber, rightPoints[j] ) , left, &minRatio, &rightIndex );
            if ( minRatio < m_AllowablePointMatchingRatio || boost::math::isinf(minRatio) ) 
            {
              MITK_WARN << "Ambiguous point match or infinite match Ratio at right frame " << frameNumber << " point " << j << " discarding point from triangulation errors"; 
            }
            else
            {
              if ( rightIndex == index ) 
              {
                matchedPairs.push_back( std::pair < unsigned int , std::pair < cv::Point2d , cv::Point2d > >
                   (index, std::pair <cv::Point2d, cv::Point2d> ( leftPoints[i], rightPoints[j] )));
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
            m_PointsInLeftLensCS[frameNumber].second[matchedPairs[i].first].first);
        
        cv::Mat leftCameraToWorld = trackerMatcher->GetCameraTrackingMatrix(frameNumber, NULL, m_TrackerIndex, NULL, m_ReferenceIndex);
      
        classifiedPoints[matchedPairs[i].first].push_back(leftCameraToWorld * triangulatedGS);
        cvReleaseMat (&leftScreenPointsMat);
        cvReleaseMat (&rightScreenPointsMat);
        cvReleaseMat (&rightScreenPointsMat);
      }
    }
    else 
    {
      MITK_WARN << "Rejecting triangulation error at frame " << frameNumber << " due to high timing error " << m_PointsInLeftLensCS[frameNumber].first << " > "  << m_AllowableTimingError ;
    }
  } 
  for ( unsigned int i = 0 ; i < classifiedPoints.size() ; i ++ ) 
  {
    MITK_INFO << "Point " << i << " triangulated mean " << mitk::GetCentroid (classifiedPoints[i],true);
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
void PickPointsOnStereoVideo::CalculateProjectionErrors (std::string outPrefix)
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
void PickPointsOnStereoVideo::CalculateReProjectionError ( std::pair < unsigned int, cv::Point2d > GSPoint, bool left )
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
  if ( abs (m_PointsInLeftLensCS[GSPoint.first].first) > m_AllowableTimingError ) 
  {
    MITK_WARN << "High timing error at " << side << " frame " << GSPoint.first << " discarding point from re-projection errors";
    return;
  }

  if ( minRatio < m_AllowablePointMatchingRatio ) 
  {
    MITK_WARN << "Ambiguous point match at " << side  << "frame "  << GSPoint.first << " discarding point from re-projection errors"; 
    return;
  }

  if ( boost::math::isinf(minRatio) ) 
  {
    MITK_WARN << "Infinite match ratio at " << side  << "frame "  << GSPoint.first << " discarding point from re-projection errors"; 
    return;
  }


  cv::Point3d matchingPointInLensCS = m_PointsInLeftLensCS[GSPoint.first].second[*index].first;

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
    mitk::UndistortPoint (GSPoint.second, *m_LeftIntrinsicMatrix, 
        *m_LeftDistortionVector, undistortedPoint,
        cropUndistortedPointsToScreen , 
        0.0, m_VideoWidth, 0.0, m_VideoHeight,cropValue);
    reProjectionGS = mitk::ReProjectPoint (undistortedPoint , *m_LeftIntrinsicMatrix);
  }
  else
  {
    cv::Point2d undistortedPoint;
    mitk::UndistortPoint (GSPoint.second, *m_RightIntrinsicMatrix, 
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
void PickPointsOnStereoVideo::CalculateProjectionError ( std::pair < unsigned int, cv::Point2d > GSPoint, bool left )
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
 
  if ( abs (m_PointsInLeftLensCS[GSPoint.first].first) > m_AllowableTimingError ) 
  {
    MITK_WARN << "High timing error at " << side << "  frame " << GSPoint.first << " discarding point from projection errors";
    return;
  }


  if ( minRatio < m_AllowablePointMatchingRatio ) 
  {
    MITK_WARN << "Ambiguous point match at " << side << "frame " << GSPoint.first << " discarding point from projection errors"; 
    return;
  }
  
  if ( boost::math::isinf(minRatio) ) 
  {
    MITK_WARN << "Infinite match ratio at " << side  << "frame "  << GSPoint.first << " discarding point from projection errors"; 
    return;
  }


  if ( left ) 
  {
    m_LeftProjectionErrors.push_back(matchingPoint - GSPoint.second);
  }
  else
  {
    m_RightProjectionErrors.push_back(matchingPoint - GSPoint.second);
  }

}

//-----------------------------------------------------------------------------
cv::Point2d PickPointsOnStereoVideo::FindNearestScreenPoint ( std::pair < unsigned int, cv::Point2d> GSPoint, bool left , double* minRatio, unsigned int* index)
{
  assert ( m_ClassifierProjectedPoints[GSPoint.first].second.size() ==
    m_ProjectedPoints[GSPoint.first].second.size() );
  std::vector < cv::Point2d > pointVector;
  for ( unsigned int i = 0 ; i < m_ClassifierProjectedPoints[GSPoint.first].second.size() ; i ++ )
  {
    if ( left )
    {
      pointVector.push_back ( m_ClassifierProjectedPoints[GSPoint.first].second[i].first );
    }
    else
    {
      pointVector.push_back ( m_ClassifierProjectedPoints[GSPoint.first].second[i].second );
    }
  }
  unsigned int myIndex;
  if ( ! boost::math::isinf(mitk::FindNearestPoint( GSPoint.second , pointVector ,minRatio, &myIndex ).x))
  {
    if ( index != NULL ) 
    {
      *index = myIndex;
    }
    if ( left ) 
    {
      return m_ProjectedPoints[GSPoint.first].second[myIndex].first;
    }
    else
    {
      return m_ProjectedPoints[GSPoint.first].second[myIndex].second;
    }
  }
  else
  {
    return cv::Point2d ( std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity() ) ;
  }
}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetWorldPoints ( 
    std::vector < std::pair < cv::Point3d , cv::Scalar > > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_WorldPoints.push_back(points[i]);
  }
  m_ProjectOK = false;
}
//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetClassifierWorldPoints ( 
    std::vector < cv::Point3d > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    m_ClassifierWorldPoints.push_back(points[i]);
  }
  m_ProjectOK = false;
}
//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetWorldPoints ( 
    std::vector < cv::Point3d > points )
{
  for ( unsigned int i = 0 ; i < points.size() ; i ++ ) 
  {
    std::pair < cv::Point3d, cv::Scalar > point = 
      std::pair < cv::Point3d , cv::Scalar > ( points[i], cv::Scalar (255,0,0) );
    m_WorldPoints.push_back(point);
  }
  m_ProjectOK = false;
}

//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::SetWorldPointsByTriangulation
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
    long long timingError;
    point =  trackerMatcher->GetCameraTrackingMatrix(framenumber[i] , &timingError , m_TrackerIndex, perturbation, m_ReferenceIndex) * point;
    if ( abs(timingError) < m_AllowableTimingError )
    {
      m_WorldPoints.push_back ( point );
      MITK_INFO << framenumber[i] << " " << onScreenPointPairs[i].first << ","
        << onScreenPointPairs[i].second << " => " << point.first << " => " << m_WorldPoints[i+wpSize].first;
    }
    else
    {
      MITK_WARN << framenumber[i] << "Point rejected due to excessive timing error: " << timingError << " > " << m_AllowableTimingError;
    }


  }
  cvReleaseMat (&leftCameraTriangulatedWorldPoints);
  m_ProjectOK = false;
}

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

std::vector < std::pair < long long , std::vector <std::pair <cv::Point2d, cv::Point2d> > > > PickPointsOnStereoVideo::GetProjectedPoints()
{
  return m_ProjectedPoints;
}
void PickPointsOnStereoVideo::ProjectAxes()
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
//-----------------------------------------------------------------------------
void PickPointsOnStereoVideo::ClearWorldPoints()
{
  m_WorldPoints.clear();
  m_ProjectOK = false;
}


} // end namespace
