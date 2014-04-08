/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkHandeyeCalibrate.h"
#include "mitkCameraCalibrationFacade.h"
#include <mitkOpenCVMaths.h>
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <niftkFileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrate::HandeyeCalibrate()
: m_FlipTracking(false)
, m_FlipExtrinsic(false)
, m_SortByDistance(true)
, m_SortByAngle(false)
, m_DoGridToWorld(true)
, m_CameraToMarker(cvCreateMat(4,4,CV_64FC1))
, m_GridToWorld(cvCreateMat(4,4,CV_64FC1))
, m_OutputDirectory("")
{
}


//-----------------------------------------------------------------------------
HandeyeCalibrate::~HandeyeCalibrate()
{

}


//-----------------------------------------------------------------------------
void HandeyeCalibrate::SetOutputDirectory(const std::string& outputDir)
{
  m_OutputDirectory = outputDir;
  this->Modified();
}


//-----------------------------------------------------------------------------
std::vector<double> HandeyeCalibrate::Calibrate(const std::string& TrackingFileDirectory,
  const std::string& ExtrinsicFileDirectoryOrFile,
  const std::string GroundTruthSolution)
{

  std::vector<cv::Mat> MarkerToWorld = mitk::LoadMatricesFromDirectory(TrackingFileDirectory);
  std::vector<cv::Mat> GridToCamera;
  std::vector<double> residuals;
  //init residuals with negative number to stop unit test passing
  //if Load result and calibration both produce zero.
  residuals.push_back(-100.0);
  residuals.push_back(-100.0);

  if ( niftk::DirectoryExists ( ExtrinsicFileDirectoryOrFile ))
  {
    GridToCamera = mitk::LoadOpenCVMatricesFromDirectory(ExtrinsicFileDirectoryOrFile);
  }
  else
  {
    GridToCamera = mitk::LoadMatricesFromExtrinsicFile(ExtrinsicFileDirectoryOrFile);
  }

  if ( MarkerToWorld.size() != GridToCamera.size() )
  {
    std::cerr << "ERROR: Called HandeyeCalibrate with unequal number of views and tracking matrices" << std::endl;
    return residuals;
  }
  int NumberOfViews = MarkerToWorld.size();
 
  std::string outputDirectory = m_OutputDirectory;
  if (outputDirectory.length() != 0)
  {
    outputDirectory = outputDirectory.append("/");
  }
  
  if ( m_FlipTracking )
  {
    MarkerToWorld = mitk::FlipMatrices(MarkerToWorld);
  }
  if ( m_FlipExtrinsic )
  {
    GridToCamera = mitk::FlipMatrices(GridToCamera);
  }

  std::vector<int> indexes;
  //if SortByDistance and SortByAngle are both true, we'll sort by distance only
  if ( m_SortByDistance )
  {
    indexes = mitk::SortMatricesByDistance(MarkerToWorld);
    std::cout << "Sorted by distances " << std::endl;
  }
  else
  {
    if ( m_SortByAngle )
    {
      indexes = mitk::SortMatricesByAngle(MarkerToWorld);
      std::cout << "Sorted by angles " << std::endl;
    }
    else
    {
      for ( unsigned int i = 0; i < MarkerToWorld.size(); i ++ )
      {
        indexes.push_back(i);
      }
      std::cout << "No Sorting" << std::endl;
    }
  }

  for ( unsigned int i = 0; i < indexes.size(); i++ )
  {
    std::cout << indexes[i] << " ";
  }
  std::cout << std::endl;

  std::vector<cv::Mat> SortedGridToCamera;
  std::vector<cv::Mat> SortedMarkerToWorld;

  for ( unsigned int i = 0; i < indexes.size(); i ++ )
  {
    SortedGridToCamera.push_back(GridToCamera[indexes[i]]);
    SortedMarkerToWorld.push_back(MarkerToWorld[indexes[i]]);
  }
 
  cv::Mat gridToWorld = cvCreateMat(4,4,CV_64FC1);
  m_CameraToMarker = HandeyeRotationAndTranslation(SortedMarkerToWorld, SortedGridToCamera,
      residuals, &gridToWorld);
  
  std::cout << "Camera To Marker Matrix = " << std::endl << m_CameraToMarker << std::endl;
  std::cout << "Rotational Residual = " << residuals [0] << std::endl;
  std::cout << "Translational Residual = " << residuals [1] << std::endl;
  std::cout << "Output directory = " << outputDirectory << std::endl;
  
  std::ofstream handeyeStream;
  handeyeStream.open((outputDirectory + "calib.left.handeye.txt").c_str());
  if ( handeyeStream ) 
  {
    for ( int i = 0 ; i < 4 ; i ++ ) 
    {
      for ( int j = 0 ; j < 4 ; j ++ )
      {
        handeyeStream << m_CameraToMarker.at<double>(i,j) << " ";
      }
      handeyeStream << std::endl;
    }
  }
  handeyeStream.close();
  if ( m_DoGridToWorld ) 
  {
    m_GridToWorld = gridToWorld;
    MITK_INFO << "Average Grid to World Transform" << std::endl << m_GridToWorld;
    std::ofstream gridCornersStream;
    gridCornersStream.open((outputDirectory + "calib.gridcorners.txt").c_str());
    if ( gridCornersStream )
    {
      for ( int i = 0 ; i < 2 ; i ++ ) 
      {
        for ( int j = 0 ; j < 2 ; j ++ ) 
        {  
          cv::Point3d x = cv::Point3d (i*(27.0) , j * (39.0), 0.0 );
          cv::Point3d y = m_GridToWorld * x;
          gridCornersStream << y.x << " " << y.y << " " << y.z << std::endl ;
        }
      }
      gridCornersStream.close();
    }
  }
  if ( GroundTruthSolution.length() > 0  )
  {
    std::vector<double> ResultResiduals;
    cv::Mat ResultMatrix = cvCreateMat(4,4,CV_64FC1);
    mitk::LoadResult(GroundTruthSolution, ResultMatrix, ResultResiduals);
    residuals[0] -= ResultResiduals[0];
    residuals[1] -= ResultResiduals[1];
    cv::Scalar Sum = cv::sum(m_CameraToMarker - ResultMatrix);
    residuals.push_back(Sum[0]);
  }

  return residuals;

}
 
} // end namespace
