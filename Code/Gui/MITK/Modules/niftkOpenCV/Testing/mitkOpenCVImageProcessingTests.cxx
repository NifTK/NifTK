/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkOpenCVImageProcessing.h>
#include <mitkOpenCVMaths.h>
#include <cmath>

/**
 * \file Tests for some of the functions in openCVImageProcessing.
 */

void FindCrossHairTest()
{

  int cannyLowThreshold = 20;
  int cannyHighThreshold = 70;
  int cannyKernel = 3;
  double houghRho = 1.0;
  double houghTheta = CV_PI/180;
  int houghThreshold = 50 ;
  int houghLineLength = 10;
  int houghLineGap = 20 ;
  cv::vector <cv::Vec4i> lines;

  cv::Mat image ( 100 , 100 , CV_8UC3 );
  for ( unsigned int i = 0 ; i < 100 ; i ++ )
  {
    for ( unsigned int j = 0 ; j < 100 ; j ++ )
    {
      for ( unsigned int channel = 0 ; channel < 3 ; channel ++ )
      {
        image.at<cv::Vec3b>(i,j)[channel] = 0;
      }
    }
  }
 
  cv::line ( image, cvPoint ( 0 , 0 ) , cvPoint ( 100 , 100 ), cv::Scalar (255,  255, 255 ),1.0 ,1 );
  cv::line ( image, cvPoint ( 100 , 10 ) , cvPoint ( 10 , 100 ), cv::Scalar (255, 255 , 255 ),1.0 ,1 );

  cv::Point2d intersect = mitk::FindCrosshairCentre (image, cannyLowThreshold, cannyHighThreshold,
      cannyKernel, houghRho, houghTheta, houghThreshold, houghLineLength, houghLineGap , lines ); 
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual (intersect,cv::Point2d (55 , 55),0.6) , "Testing intersect for no noise state" << intersect );
}

void ApplyMaskTest()
{
  std::vector < std::pair < cv::Point2d, cv::Point2d > > pointPairs;
  cv::Mat mask;
  unsigned int maskValue = 0;
  
  MITK_TEST_CONDITION ( mitk::ApplyMask ( pointPairs, mask , maskValue, true ) == 0 , "Testing apply mask works on empty vectors ");

  mask = cv::Mat::zeros ( 4, 4, CV_8U);
  for ( unsigned int i = 2 ; i < 4 ; i ++ )
  {
    for ( unsigned int j = 2 ; j < 4 ; j ++ ) 
    {
      mask.at<unsigned int> (i,j) = 255;
    }
  }
  
  pointPairs.push_back ( std::pair <cv::Point2d, cv::Point2d > ( cv::Point2d (0,0), cv::Point2d(0,0) ) ); //both out of mask
  pointPairs.push_back ( std::pair <cv::Point2d, cv::Point2d > ( cv::Point2d (-1,0), cv::Point2d(0,0) ) ); //left out in bounds
  pointPairs.push_back ( std::pair <cv::Point2d, cv::Point2d > ( cv::Point2d (3,3), cv::Point2d(0,-56.5) ) ); //right out in bounds
  pointPairs.push_back ( std::pair <cv::Point2d, cv::Point2d > ( cv::Point2d (2,3), cv::Point2d(3,3) ) ); //both in mask
  pointPairs.push_back ( std::pair <cv::Point2d, cv::Point2d > ( cv::Point2d (1.7,3.2), cv::Point2d(3.5,3.0) ) ); //both in mask ?

  MITK_TEST_CONDITION ( mitk::ApplyMask ( pointPairs , mask , maskValue , true ) == 2 , "Testing apply mask removed 2 left pointpairs");
  MITK_TEST_CONDITION ( mitk::ApplyMask ( pointPairs , mask , maskValue , false ) == 1 , "Testing apply mask removed 1 right pointpairs");

}


int mitkOpenCVImageProcessingTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVImageProcessingTests");

  FindCrossHairTest();
  ApplyMaskTest();
  MITK_TEST_END();
}



