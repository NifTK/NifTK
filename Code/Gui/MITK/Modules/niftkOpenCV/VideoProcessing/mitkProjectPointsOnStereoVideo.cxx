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
  
  if ( m_TrackerMatcher == NULL  )
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
  while ( framenumber < m_TrackerMatcher->GetNumberOfFrames() )
  {
    //put the world points into the coordinates of the left hand camera.
    //worldtotracker * trackertocamera
    //in general the tracker matrices are trackertoworld
    cv::Mat WorldToLeftCamera = 
      m_TrackerMatcher->GetTrackerMatrix(framenumber,NULL, m_TrackerIndex).inv() 
      * m_LeftCameraToTracker->inv();
   
   // m_PointsInLeftLensCS = TransformPoints(WorldToLeftCamera , m_WorldPoints);
    m_PointsInLeftLensCS = WorldToLeftCamera * m_WorldPoints;

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
                                          


} // end namespace
