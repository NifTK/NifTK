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
#include "FileHelper.h"

namespace mitk {

//-----------------------------------------------------------------------------
HandeyeCalibrate::HandeyeCalibrate()
{

}


//-----------------------------------------------------------------------------
HandeyeCalibrate::~HandeyeCalibrate()
{

}


//-----------------------------------------------------------------------------
cv::Mat HandeyeCalibrate::Calibrate(const std::vector<cv::Mat>  MarkerToWorld, 
    const std::vector<cv::Mat> GridToCamera, std::vector<double>* residuals)
{
  if ( MarkerToWorld.size() != GridToCamera.size() )
  {
    std::cerr << "ERROR: Called HandeyeCalibrate with unequal number of views and tracking matrices" << std::endl;
    cv::Mat empty = cvCreateMat(0,0,CV_64FC1);
    return empty;
  }
  int NumberOfViews = MarkerToWorld.size();
  
  cv::Mat A = cvCreateMat ( 3 * (NumberOfViews - 1), 3, CV_64FC1 );
  cv::Mat b = cvCreateMat ( 3 * (NumberOfViews - 1), 1, CV_64FC1 );

  for ( int i = 0 ; i < NumberOfViews - 1 ; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = MarkerToWorld[i+1].inv() * MarkerToWorld[i];
    mat2 = GridToCamera[i+1] * GridToCamera[i].inv();

    cv::Mat rotationMat1 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationMat2 = cvCreateMat(3,3,CV_64FC1);
    cv::Mat rotationVector1 = cvCreateMat(3,1,CV_64FC1);
    cv::Mat rotationVector2 = cvCreateMat(3,1,CV_64FC1);
    for ( int row = 0 ; row < 3 ; row ++ )
    {
      for ( int col = 0 ; col < 3 ; col ++ )
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
  if ( residuals != NULL )
  {
    residuals->push_back(RotationResidual);
  }
  
  cv::Mat pcg = 2 * pcgPrime / ( sqrt(1 + cv::norm(pcgPrime) * cv::norm(pcgPrime)) );
  cv::Mat id3 = cvCreateMat(3,3,CV_64FC1);
  for ( int row = 0 ; row < 3 ; row ++ )
  {
    for ( int col = 0 ; col < 3 ; col ++ )
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
    + 0.5 * ( pcg_mulTransposed + sqrt(4 - norm(pcg) * norm(pcg))*pcg_crossproduct) ;

  //now do the translation
  for ( int i = 0 ; i < NumberOfViews - 1 ; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_64FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_64FC1);
    mat1 = MarkerToWorld[i+1].inv() * MarkerToWorld[i];
    mat2 = GridToCamera[i+1] * GridToCamera[i].inv();

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
    for ( int j = 0 ; j < 3 ; j ++ ) 
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
  if ( residuals != NULL )
  {
    residuals->push_back(TransResidual);
  }

  cv::Mat CameraToMarker = cvCreateMat(4,4,CV_64FC1);
  for ( int row = 0 ; row < 3 ; row ++ ) 
  {
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      CameraToMarker.at<double>(row,col) = rcg.at<double>(row,col);
    }
  }
  for ( int row = 0 ; row < 3 ; row ++ )
  {
    CameraToMarker.at<double>(row,3) = tcg.at<double>(row,0);
  }
  for ( int col = 0 ; col < 3 ; col ++ )
  {
    CameraToMarker.at<double>(3,col) = 0.0;
  }
  CameraToMarker.at<double>(3,3)=1.0;
  return CameraToMarker;

}
  
} // end namespace
