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
double HandeyeCalibrate::Calibrate(const std::vector<cv::Mat>  MarkerToWorld, 
    const std::vector<cv::Mat> GridToCamera,
    cv::Mat CameraToMarker
    )
{
  if ( MarkerToWorld.size() != GridToCamera.size() )
  {
    std::cerr << "ERROR: Called HandeyeCalibrate with unequal number of views and tracking matrices" << std::endl;
    return 0.0;
  }
  int NumberOfViews = MarkerToWorld.size();
  
  cv::Mat A = cvCreateMat ( 3 * NumberOfViews, 3, CV_32FC1 );
  cv::Mat b = cvCreateMat ( 3 * NumberOfViews, 1, CV_32FC1 );

  for ( int i = 0 ; i < NumberOfViews - 1 ; i ++ )
  {
    cv::Mat mat1 = cvCreateMat(4,4,CV_32FC1);
    cv::Mat mat2 = cvCreateMat(4,4,CV_32FC1);
    mat1 = MarkerToWorld[i+1].inv() * MarkerToWorld[i];
    mat2 = GridToCamera[i+1] * GridToCamera[i].inv();

    cv::Mat rotationMat1 = cvCreateMat(3,3,CV_32FC1);
    cv::Mat rotationMat2 = cvCreateMat(3,3,CV_32FC1);
    cv::Mat rotationVector1 = cvCreateMat(3,1,CV_32FC1);
    cv::Mat rotationVector2 = cvCreateMat(3,1,CV_32FC1);
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

    rotationVector1 *= 2*sin(norm1/2);
    rotationVector2 *= 2*sin(norm2/2);

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
  
  cv::Mat PseudoInverse = cvCreateMat(3,3,CV_32FC1);
  cv::invert(A,PseudoInverse,CV_SVD);
  return 0.0;
}
} // end namespace
