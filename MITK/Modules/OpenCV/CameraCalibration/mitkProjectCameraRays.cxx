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
, m_UndistortBeforeProjection(true)
, m_ScreenWidth(1920)
, m_ScreenHeight(540)
, m_RayLength ( 500 )
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
void ProjectCameraRays::WriteOutput ( std::string filename )
{
  std::streambuf * buf;
  std::ofstream of;

  if( filename.length() != 0 )
  {

    of.open(filename);
    if ( ! of )
    {
      mitkThrow() << "Failed to open file to write projected rays " << filename;
    }
    buf = of.rdbuf();
  }
  else
  {
    MITK_WARN << "ProjectCameraRays: No output file specified, writing to standard out.";
    buf = std::cout.rdbuf();
  }

  std::ostream out(buf);

  MITK_INFO << "Writing results to " << filename;
  for ( std::vector<std::pair<cv::Point3d, cv::Point3d> > ::iterator it = m_Rays.begin() ; it < m_Rays.end() ; ++it )
  {
    out << std::setprecision(20) << it->first.x << " " << it->first.y << " " << it->first.z << " ";
    out << std::setprecision(20) << it->second.x << " " << it->second.y << " " << it->second.z  <<  std::endl;
  }
  of.close();

}

//-----------------------------------------------------------------------------
std::vector < std::pair < cv::Point3d, cv::Point3d > >  ProjectCameraRays::GetRays()
{
  return m_Rays;
}

//-----------------------------------------------------------------------------
bool ProjectCameraRays::Project()
{
  bool isSuccessful = false;

  try
  {
    //check length of screen point vector
    MITK_INFO << "Initialising screen points";

    InitScreenPointsVector();

    cv::Mat intrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat distortion = cvCreateMat (1,4,CV_64FC1);    // not used (yet)
    cv::Mat lensToWorld = cv::Mat::eye (4,4, CV_64FC1);

    if ( m_LensToWorldFileName != "" )
    {
      if ( ! ( ReadTrackerMatrix ( m_LensToWorldFileName,lensToWorld ) ) )
      {
        MITK_ERROR << "ProjectCameraRays: Failed to read matrix from " << m_LensToWorldFileName;
        isSuccessful = false;
        return isSuccessful;
      }
    }

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

    unsigned int index = 0;
    for ( std::vector<cv::Point2d>::iterator it = leftPoints_undistorted.begin() ; it < leftPoints_undistorted.end() ;  ++it)
    {
      pointsToProject.at<double>(0,index) = it->x;
      pointsToProject.at<double>(1,index) = it->y;
      pointsToProject.at<double>(2,index) = 1.0;
      ++index;
    }

    MITK_INFO << "Projecting rays";
    m_Rays = mitk::GetRays ( pointsToProject , intrinsic , m_RayLength, lensToWorld );

    isSuccessful = true;
  }
  catch (const std::logic_error& e)
  {
    std::cerr << "ProjectCameraRays::Project: exception thrown e=" << e.what() << std::endl;
  }

  if ( isSuccessful )
  {
    MITK_INFO << "Success !";
  }
  else
  {
    MITK_INFO << "Failure !";
  }
  return isSuccessful;
}

} // end namespace
