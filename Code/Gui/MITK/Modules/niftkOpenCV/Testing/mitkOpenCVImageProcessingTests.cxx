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
#include <cmath>

/**
 * \file Tests for some of the functions in openCVImageProcessing.
 */

bool FindCrossHairTest()
{
  MITK_TEST_BEGIN ("mitkOpenCVFindCrossHairTest");


  int cannyLowThreshold = 0;
  int cannyHighThreshold = 10;
  int cannyKernel = 10;
  double houghRho = 0;
  double houghTheta = 0;
  int houghThreshold = 0 ;
  int houghLineLength = 10;
  int houghLineGap = 0 ;
  cv::vector <cv::Vec4i> lines;

  cv::Mat image ( 100 , 100 , CV_64FC3 );
  cv::line ( image, cvPoint ( 60 , 40 ) , cvPoint ( 60 , 60 ), cv::Scalar ( 0, 0, 0 ) );

  cv::Point2d intersect = mitk::FindCrosshairCentre (image, cannyLowThreshold, cannyHighThreshold,
      cannyKernel, houghRho, houghTheta, houghThreshold, houghLineLength, houghLineGap , lines ); 
  
  MITK_TEST_CONDITION ( intersect == cv::Point2d (60 , 50) , "Testing intersect for no noise state" );
  MITK_TEST_END();
}


int mitkOpenCVImageProcessingTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVImageProcessingTests");

  FindCrossHairTest();
  MITK_TEST_END();
}



