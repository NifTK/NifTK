/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVTest.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

namespace mitk {

Mat src, srcGrey;
Mat dst, detectedEdges;
Mat outputHough;
int cannyBlur = 1;
int cannyLowThreshold = 100;
int cannyUpperThreshold = 150;
int houghAccumulatorThreshold = 20;
int houghMinDistance = 5;
int houghMaxGap = 5;
int kernelSize = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
  blur( srcGrey, detectedEdges, Size(cannyBlur,cannyBlur) );
  Canny( detectedEdges, detectedEdges, cannyLowThreshold, cannyUpperThreshold, kernelSize );
  imshow( window_name, detectedEdges );
 }

void HoughLines(int, void*)
{
  outputHough = src.clone();


  vector<Vec4i> p_lines;
  p_lines.clear();

  HoughLinesP( detectedEdges, p_lines, 2, CV_PI/180, houghAccumulatorThreshold, houghMinDistance, houghMaxGap );
  for( size_t i = 0; i < p_lines.size(); i++ )
     {
       Vec4i l = p_lines[i];
       line( outputHough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 1, CV_AA);
     }
/*
  vector<Vec2f> s_lines;
  HoughLines( detectedEdges, s_lines, 1, CV_PI/180, houghAccumulatorThreshold);

  for( size_t i = 0; i < s_lines.size(); i++ )
     {
      float r = s_lines[i][0], t = s_lines[i][1];
      double cos_t = cos(t), sin_t = sin(t);
      double x0 = r*cos_t, y0 = r*sin_t;
      double alpha = 1000;

       Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
       Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
       line( outputHough, pt1, pt2, Scalar(255,0,0), 1, CV_AA);
     }
*/
  imshow( window_name, outputHough );
}

//-----------------------------------------------------------------------------
OpenCVTest::OpenCVTest()
{
}


//-----------------------------------------------------------------------------
OpenCVTest::~OpenCVTest()
{

}


//-----------------------------------------------------------------------------
void OpenCVTest::Run(const std::string& fileName)
{
  /// Load data.
  src = imread( fileName );
  if( !src.data )
  {
    return;
  }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, srcGrey, CV_RGB2GRAY );

  /// Create a window
  namedWindow( window_name );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Blur", window_name, &cannyBlur, 255, CannyThreshold );
  createTrackbar( "Min Threshold:", window_name, &cannyLowThreshold, 255, CannyThreshold );
  createTrackbar( "Max Threshold:", window_name, &cannyUpperThreshold, 255, CannyThreshold );
  createTrackbar( "Hough Acc Threshold:", window_name, &houghAccumulatorThreshold, 255, HoughLines );
  createTrackbar( "Hough Min Dist:", window_name, &houghMinDistance, 255, HoughLines );
  createTrackbar( "Hough Max Gap:", window_name, &houghMaxGap, 255, HoughLines );

  /// Show the image
  CannyThreshold(0,0);
  HoughLines(0,0);

  /// Wait until user exit program by pressing a key
  waitKey(0);
}

//-----------------------------------------------------------------------------
} // end namespace
