/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectCameraRays.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <niftkFileHelper.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkOpenCVImageProcessing.h>

namespace mitk {

//-----------------------------------------------------------------------------
ProjectCameraRays::ProjectCameraRays()
: m_LensToWorldFileName("")
, m_IntrinsicFileName("")
, m_OutputFileName("")
, m_UndistortBeforeProjection(true)
, m_ScreenWidth(1920)
, m_ScreenHeight(540)
{
}

//-----------------------------------------------------------------------------
ProjectCameraRays::~ProjectCameraRays()
{
}

//-----------------------------------------------------------------------------
void ProjectCameraRays::LoadScreenPointsFromFile (std::string fileName)
{
  //to do
}

//-----------------------------------------------------------------------------
void ProjectCameraRays::InitScreenPointsVector ()
{
  if ( m_ScreenPoints.size() != 0 )
  {
    //nothing to do
    return;
  }
  else
  {
    for ( int x = 0 ; x < m_ScreenWidth ; ++x )
    {
      for ( int y = 0 ; y < m_ScreenHeight ; ++y )
      {
        m_ScreenPoints.push_back ( cv::Point2d ( static_cast<double>(x), static_cast<double>(y) ) );
      }
    }
    return;
  }
}


//-----------------------------------------------------------------------------
bool ProjectCameraRays::Project()
{
  bool isSuccessful = false;

  try
  {
    //check length of screen point vector
    InitScreenPointsVector();

    cv::Mat intrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat distortion = cvCreateMat (1,4,CV_64FC1);    // not used (yet)
    cv::Mat lensToWorld = cv::Mat::eye (4,4, CV_64FC1);

    cv::Mat pointsToProject = cv::Mat ( 3, m_ScreenPoints.size(), CV_64FC1 );

    // Load matrices. These throw exceptions if things fail.
    LoadCameraIntrinsicsFromPlainText(m_IntrinsicFileName, &intrinsic, &distortion);

    std::vector<cv::Point2d> leftPoints_undistorted;
    if ( m_UndistortBeforeProjection )
    {
      mitk::UndistortPoints(m_ScreenPoints, intrinsic, distortion, leftPoints_undistorted);
    }
    else
    {
      leftPoints_undistorted = m_ScreenPoints;
    }

   // mitk::GetRays
    isSuccessful = true;
  }
  catch (const std::logic_error& e)
  {
    std::cerr << "ProjectCameraRays::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
