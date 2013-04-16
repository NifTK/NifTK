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
  std::cout << RotationResidual << std::endl;
  
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
  std::cout << pcg_mulTransposed;
  cv::Mat rcg = ( 1 - cv::norm(pcg) * norm(pcg) /2 ) * id3 ;
  std::cout << rcg << std::endl;
        
  return 0.0;
}
  
//-----------------------------------------------------------------------------
std::vector<cv::Mat> HandeyeCalibrate::LoadMatricesFromDirectory (const std::string& fullDirectoryName)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  std::sort(files.begin(),files.end());
  std::vector<cv::Mat> myMatrices;

  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
      std::ifstream fin(files[i].c_str());
      std::cout << "Reading Matrix from " << files[i].c_str() << std::endl;
      for ( int row = 0 ; row < 4 ; row ++ ) 
      {
        for ( int col = 0 ; col < 4 ; col ++ ) 
        {
          fin >> Matrix.at<double>(row,col);
        }
      }
      myMatrices.push_back(Matrix);
    }
  }
  else
  {
    throw std::logic_error("No files found in directory!");
  }

  if (myMatrices.size() == 0)
  {
    throw std::logic_error("No images found in directory!");
  }
  std::cout << "Loaded " << myMatrices.size() << " Matrices from " << fullDirectoryName << std::endl;
  return myMatrices;
}

//-----------------------------------------------------------------------------
std::vector<cv::Mat> HandeyeCalibrate::LoadMatricesFromExtrinsicFile (const std::string& fullFileName)
{

  std::vector<cv::Mat> myMatrices;
  std::ifstream fin(fullFileName.c_str());

  cv::Mat RotationVector = cvCreateMat(3,1,CV_64FC1);
  cv::Mat TranslationVector = cvCreateMat(3,1,CV_64FC1);
  double temp_d[6];
  while ( fin >> temp_d[0] >> temp_d[1] >> temp_d[2] >> temp_d[3] >> temp_d[4] >> temp_d[5] )
  {
    RotationVector.at<double>(0,0) = temp_d[0];
    RotationVector.at<double>(1,0) = temp_d[1];
    RotationVector.at<double>(2,0) = temp_d[2];
    TranslationVector.at<double>(0,0) = temp_d[3];
    TranslationVector.at<double>(1,0) = temp_d[4];
    TranslationVector.at<double>(2,0) = temp_d[5];
    
    cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
    cv::Mat RotationMatrix = cvCreateMat(3,3,CV_64FC1);
    cv::Rodrigues (RotationVector, RotationMatrix);

    for ( int row = 0 ; row < 3 ; row ++ ) 
    {
      for ( int col = 0 ; col < 3 ; col ++ ) 
      {
        Matrix.at<double>(row,col) = RotationMatrix.at<double>(row,col);
      }
    }
   
    for ( int row = 0 ; row < 3 ; row ++ )
    {
      Matrix.at<double>(row,3) = TranslationVector.at<double>(row,0);
    }
    for ( int col = 0 ; col < 3 ; col ++ ) 
    {
      Matrix.at<double>(3,col) = 0.0;
    }
    Matrix.at<double>(3,3) = 1.0;
    myMatrices.push_back(Matrix);
  }
  return myMatrices;
}

//-----------------------------------------------------------------------------
std::vector<cv::Mat> HandeyeCalibrate::FlipMatrices (const std::vector<cv::Mat> Matrices)
{
  
  std::vector<cv::Mat>  OutMatrices;
  for ( unsigned int i = 0 ; i < Matrices.size() ; i ++ )
  {
    cv::Mat FlipMat = cvCreateMat(4,4,CV_64FC1);
    FlipMat.at<double>(0,0) = Matrices[i].at<double>(0,0);
    FlipMat.at<double>(0,1) = Matrices[i].at<double>(0,1);
    FlipMat.at<double>(0,2) = Matrices[i].at<double>(0,2) * -1;
    FlipMat.at<double>(0,3) = Matrices[i].at<double>(0,3);

    FlipMat.at<double>(1,0) = Matrices[i].at<double>(1,0);
    FlipMat.at<double>(1,1) = Matrices[i].at<double>(1,1);
    FlipMat.at<double>(1,2) = Matrices[i].at<double>(1,2) * -1;
    FlipMat.at<double>(1,3) = Matrices[i].at<double>(1,3);

    FlipMat.at<double>(2,0) = Matrices[i].at<double>(2,0) * -1;
    FlipMat.at<double>(2,1) = Matrices[i].at<double>(2,1) * -1;
    FlipMat.at<double>(2,2) = Matrices[i].at<double>(2,2);
    FlipMat.at<double>(2,3) = Matrices[i].at<double>(2,3) * -1;

    FlipMat.at<double>(3,0) = Matrices[i].at<double>(3,0);
    FlipMat.at<double>(3,1) = Matrices[i].at<double>(3,1);
    FlipMat.at<double>(3,2) = Matrices[i].at<double>(3,2);
    FlipMat.at<double>(3,3) = Matrices[i].at<double>(3,3);

    OutMatrices.push_back(FlipMat);
  }
  return OutMatrices;
}

} // end namespace
