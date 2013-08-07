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
