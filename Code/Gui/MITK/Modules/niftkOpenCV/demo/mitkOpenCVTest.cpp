/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

//-----------------------------------------------------------------------------
OpenCVTest::OpenCVTest()
{

}


//-----------------------------------------------------------------------------
OpenCVTest::~OpenCVTest()
{

}

void detectAndDisplay( Mat frame )
{
 std::vector<Rect> faces;
 Mat frame_gray;
 Mat standard_hough;
 Mat probabilistic_hough;
 Mat edges;

 cvtColor( frame, frame_gray, CV_BGR2GRAY );

 vector<Vec4i> p_lines;
 Canny( frame_gray, edges, 50, 150, 3 );
 HoughLinesP( edges, p_lines, 1, CV_PI/180, 100, 50, 50 );
 for( size_t i = 0; i < p_lines.size(); i++ )
    {
      Vec4i l = p_lines[i];
      line( frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 1, CV_AA);
    }


// GaussianBlur(frame_gray, frame_gray, Size(9,9), 2, 2);
 vector<Vec3f> circles;
 HoughCircles( frame_gray, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows/8, 150, 25, 10, 50 );
 for( size_t i = 0; i < circles.size(); i++ )
 {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( frame, center, radius, Scalar(0,0,255), 1, 8, 0 );
 }


/*
 vector<Vec2f> s_lines;
 Canny( frame_gray, edges, 50, 150, 3 );
 //cvtColor( edges, standard_hough, CV_GRAY2BGR );
 HoughLines( edges, s_lines, 1, CV_PI/180, 100, 2, 2 );

 for( size_t i = 0; i < s_lines.size(); i++ )
    {
     float r = s_lines[i][0], t = s_lines[i][1];
     double cos_t = cos(t), sin_t = sin(t);
     double x0 = r*cos_t, y0 = r*sin_t;
     double alpha = 1000;

      Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
      Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
      line( frame, pt1, pt2, Scalar(255,0,0), 1, CV_AA);
    }
*/

 /*
 //-- Detect faces
 equalizeHist( frame_gray, frame_gray );
 face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

 for( int i = 0; i < faces.size(); i++ )
 {
   Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
   ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

   Mat faceROI = frame_gray( faces[i] );
   std::vector<Rect> eyes;

   //-- In each face, detect eyes
   eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

   for( int j = 0; j < eyes.size(); j++ )
    {
      Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
      int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
      circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
    }
 }
 */


 //-- Show what you got
 imshow( "Example2", frame );
}

/** @function main */
void OpenCVTest::Run(const std::string& fileName)
{
  cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
  CvCapture* capture = NULL;
  Mat frame;

  if (fileName.length() == 0)
  {
    capture = cvCreateCameraCapture(-1);
  }
  else
  {
    capture = cvCreateFileCapture(fileName.c_str());
  }

  if (capture)
  {
    while(1)
    {
      frame = cvQueryFrame(capture);
      if (frame.empty())
      {
        printf(" --(!) No captured frame -- Break!\n");
        break;
      }
      else
      {
        detectAndDisplay( frame );
      }
      int c = cvWaitKey(10);
      if( (char)c == 'c' ) { break; }
    }
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example2");
  return;
}

//-----------------------------------------------------------------------------
} // end namespace
