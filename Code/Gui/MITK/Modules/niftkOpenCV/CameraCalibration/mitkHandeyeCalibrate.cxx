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
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <FileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrate::HandeyeCalibrate()
: m_FlipTracking(true)
, m_FlipExtrinsic(false)
, m_SortByDistance(false)
, m_SortByAngle(false)
{

}


//-----------------------------------------------------------------------------
HandeyeCalibrate::~HandeyeCalibrate()
{

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

  cv::Mat A = cvCreateMat ( 3 * (NumberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (NumberOfViews - 1), 1, CV_64FC1 );

  for ( int i = 0; i < NumberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = SortedMarkerToWorld[i+1].inv() * SortedMarkerToWorld[i];
    mat2 = SortedGridToCamera[i+1] * SortedGridToCamera[i].inv();

    cv::Mat rotationMat1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationMat2 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationVector1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat rotationVector2 = cvCreateMat(3,1,CV_64FC1);
    for ( int row = 0; row < 3; row ++ )
    {
      for ( int col = 0; col < 3; col ++ )
      {
        rotationMat1.at<double>(row,col) = mat1.at<double>(row,col);
        rotationMat2.at<double>(row,col) = mat2.at<double>(row,col);
      }
    }
    cv::Rodrigues (rotationMat1, rotationVector1 );
    cv::Rodrigues (rotationMat2, rotationVector2 );

    double norm1 = cv::norm(rotationVector1);
    double norm2 = cv::norm(rotationVector2);

    rotationVector1 *= 2*sin(norm1/2) / norm1;
    rotationVector2 *= 2*sin(norm2/2) / norm2;

    cv::Mat sum = rotationVector1 + rotationVector2;
    cv::Mat diff = rotationVector2 - rotationVector1;

    A.at<double>(i*3+0,0)=0.0;
    A.at<double>(i*3+0,1)=-(sum.at<double>(2,0));
    A.at<double>(i*3+0,2)=sum.at<double>(1,0);
    A.at<double>(i*3+1,0)=sum.at<double>(2,0);
    A.at<double>(i*3+1,1)=0.0;
    A.at<double>(i*3+1,2)=-(sum.at<double>(0,0));
    A.at<double>(i*3+2,0)=-(sum.at<double>(1,0));
    A.at<double>(i*3+2,1)=sum.at<double>(0,0);
    A.at<double>(i*3+2,2)=0.0;
 
    b.at<double>(i*3+0,0)=diff.at<double>(0,0);
    b.at<double>(i*3+1,0)=diff.at<double>(1,0);
    b.at<double>(i*3+2,0)=diff.at<double>(2,0);
  
  }
  
  cv::Mat PseudoInverse = cvCreateMat(3,3,CV_64FC1);
  cv::invert(A,PseudoInverse,CV_SVD);
 
  cv::Mat pcgPrime = PseudoInverse * b;

  cv::Mat Error = A * pcgPrime-b;
 
  cv::Mat ErrorTransMult = cvCreateMat(Error.cols, Error.cols, CV_64FC1);
 
  cv::mulTransposed (Error, ErrorTransMult, true);
     
  double RotationResidual = sqrt(ErrorTransMult.at<double>(0,0)/(NumberOfViews-1));
  residuals[0] = RotationResidual;
 
  cv::Mat pcg = 2 * pcgPrime / ( sqrt(1 + cv::norm(pcgPrime) * cv::norm(pcgPrime)) );
  cv::Mat id3 = cvCreateMat(3,3,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      if ( row == col )
      {
        id3.at<double>(row,col) = 1.0;
      }
      else
      {
        id3.at<double>(row,col) = 0.0;
      }
    }
  }
      
  cv::Mat pcg_crossproduct = cvCreateMat(3,3,CV_64FC1);
  pcg_crossproduct.at<double>(0,0)=0.0;
  pcg_crossproduct.at<double>(0,1)=-(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(0,2)=(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(1,0)=(pcg.at<double>(2,0));
  pcg_crossproduct.at<double>(1,1)=0.0;
  pcg_crossproduct.at<double>(1,2)=-(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,0)=-(pcg.at<double>(1,0));
  pcg_crossproduct.at<double>(2,1)=(pcg.at<double>(0,0));
  pcg_crossproduct.at<double>(2,2)=0.0;
 
  cv::Mat pcg_mulTransposed = cvCreateMat(pcg.rows, pcg.rows, CV_64FC1);
  cv::mulTransposed (pcg, pcg_mulTransposed, false);
  cv::Mat rcg = ( 1 - cv::norm(pcg) * norm(pcg) /2 ) * id3
    + 0.5 * ( pcg_mulTransposed + sqrt(4 - norm(pcg) * norm(pcg))*pcg_crossproduct);

  //now do the translation
  for ( int i = 0; i < NumberOfViews - 1; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = SortedMarkerToWorld[i+1].inv() * SortedMarkerToWorld[i];
    mat2 = SortedGridToCamera[i+1] * SortedGridToCamera[i].inv();

    A.at<double>(i*3+0,0)=mat1.at<double>(0,0) - 1.0;
    A.at<double>(i*3+0,1)=mat1.at<double>(0,1) - 0.0;
    A.at<double>(i*3+0,2)=mat1.at<double>(0,2) - 0.0;
    A.at<double>(i*3+1,0)=mat1.at<double>(1,0) - 0.0;
    A.at<double>(i*3+1,1)=mat1.at<double>(1,1) - 1.0;
    A.at<double>(i*3+1,2)=mat1.at<double>(1,2) - 0.0;
    A.at<double>(i*3+2,0)=mat1.at<double>(2,0) - 0.0;
    A.at<double>(i*3+2,1)=mat1.at<double>(2,1) - 0.0;
    A.at<double>(i*3+2,2)=mat1.at<double>(2,2) - 1.0;
 
    cv::Mat m1_t = cvCreateMat(3,1,CV_64FC1);
    cv::Mat m2_t = cvCreateMat(3,1,CV_64FC1);
    for ( int j = 0; j < 3; j ++ )
    {
      m1_t.at<double>(j,0) = mat1.at<double>(j,3);
      m2_t.at<double>(j,0) = mat2.at<double>(j,3);
    }
    cv::Mat b_t = rcg * m2_t - m1_t;
  
    b.at<double>(i*3+0,0)=b_t.at<double>(0,0);
    b.at<double>(i*3+1,0)=b_t.at<double>(1,0);
    b.at<double>(i*3+2,0)=b_t.at<double>(2,0);

  
  }
       
  cv::invert(A,PseudoInverse,CV_SVD);
 
  cv::Mat tcg = PseudoInverse * b;

  Error = A * tcg -b;
 
  cv::mulTransposed (Error, ErrorTransMult, true);
     
  double TransResidual = sqrt(ErrorTransMult.at<double>(0,0)/(NumberOfViews-1));
  residuals[1] = TransResidual;

  cv::Mat CameraToMarker = cvCreateMat(4,4,CV_64FC1);
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      CameraToMarker.at<double>(row,col) = rcg.at<double>(row,col);
    }
  }
  for ( int row = 0; row < 3; row ++ )
  {
    CameraToMarker.at<double>(row,3) = tcg.at<double>(row,0);
  }
  for ( int col = 0; col < 3; col ++ )
  {
    CameraToMarker.at<double>(3,col) = 0.0;
  }
  CameraToMarker.at<double>(3,3)=1.0;
  std::cout << "Camera To Marker Matrix = " << std::endl << CameraToMarker << std::endl;
  std::cout << "Rotational Residual = " << residuals [0] << std::endl;
  std::cout << "Translational Residual = " << residuals [1] << std::endl;

  if ( GroundTruthSolution.length() > 0  )
  {
    std::vector<double> ResultResiduals;
    cv::Mat ResultMatrix = cvCreateMat(4,4,CV_64FC1);
    mitk::LoadResult(GroundTruthSolution, ResultMatrix, ResultResiduals);
    residuals[0] -= ResultResiduals[0];
    residuals[1] -= ResultResiduals[1];
    cv::Scalar Sum = cv::sum(CameraToMarker - ResultMatrix);
    residuals.push_back(Sum[0]);
  }

  return residuals;

}
 
} // end namespace
