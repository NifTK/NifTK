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
, leftIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, leftDistortionVector (new cv::Mat(5,1,CV_32FC1))
, rightIntrinsicMatrix (new cv::Mat(3,3,CV_32FC1))
, rightDistortionVector (new cv::Mat(5,1,CV_32FC1))
, rightToLeftRotationMatrix (new cv::Mat(3,3,CV_32FC1))
, rightToLeftTranslationVector (new cv::Mat(3,1,CV_32FC1))
, leftCameraToTracker (new cv::Mat(4,4,CV_32FC1))
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
      leftIntrinsicMatrix,leftDistortionVector,rightIntrinsicMatrix,
      rightDistortionVector,rightToLeftRotationMatrix,
      rightToLeftTranslationVector,leftCameraToTracker);
  }
  catch ( int e )
  {
    MITK_ERROR << "Failed to load camera parameters";
    m_InitOK = false;
    return;
  }

  m_TrackerMatcher = mitk::VideoTrackerMatching::New();
  m_TrackerMatcher->Initialise(m_Directory);
  if ( ! m_TrackerMatcher->IsReady() )
  {
    MITK_ERROR << "Failed to initialise tracker matcher";
    m_InitOK = false;
    return;
  }

  if ( m_Visualise || m_SaveVideo ) 
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
  }

  m_Capture = cvCreateFileCapture(m_VideoIn.c_str()); 

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
