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

bool FindCrossHairTest()
{
  MITK_TEST_BEGIN ("mitkOpenCVFindCrossHairTest");

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
  MITK_TEST_END();
}


int mitkOpenCVImageProcessingTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVImageProcessingTests");

  FindCrossHairTest();
  MITK_TEST_END();
}



