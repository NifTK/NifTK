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

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkHandeyeCalibrate.h>
#include <mitkCameraCalibrationFacade.h>
#include <cv.h>
#include <highgui.h>


/*
 * This is going to read in a video stream showing a sequence of views of an
 * unambiguous cross hair object. We use opencv to convert the video into 
 * on screen coordinates for each channel and each frame.
 * A single frame is selected to determine the position of the point in 
 * 3D space. This point is projected into subsequent frames.
 */

cv::Mat WorldToLens (cv::Mat PointInWorldCoordinates, cv::Mat TrackerToWorld,
    cv::Mat TrackerToLens);

cv::Mat LensToWorld (cv::Mat PointInLensCoordinates, cv::Mat TrackerToWorld,
    cv::Mat TrackerToLens);


int mitkTrackingTest ( int argc, char * argv[] )
{
  std::string inputVideo = argv[1];


  //get the video and show it
  CvCapture *capture = 0 ;
  MITK_INFO << "Opening " << inputVideo;
  capture=cvCreateFileCapture(inputVideo.c_str());
  //capture=cvCreateCameraCapture(CV_CAP_V4L);
  
  if ( ! capture ) 
  {
    MITK_WARN << "Failed to open " << inputVideo;
  }
  else 
  {
    MITK_INFO << "Opened OK";
  }

  cvNamedWindow( "Left Channel",CV_WINDOW_AUTOSIZE);
  cvNamedWindow( "Right Channel",CV_WINDOW_AUTOSIZE);
  cvNamedWindow( "Left Processed",CV_WINDOW_AUTOSIZE);

  int key = 0 ;
  IplImage *leftframe;
  IplImage *rightframe;
  IplImage *smallleft;
  IplImage *smallright;

  IplImage *leftprocessed;
  IplImage *rightprocessed;
  IplImage *leftprocessed_temp;
  IplImage *rightprocessed_temp;
  IplImage *smallleftprocessed;
  IplImage *smallrightprocessed;

  smallleft = cvCreateImage( cvSize(640,360), 8, 3 );
  smallright = cvCreateImage( cvSize(640,360), 8, 3 );
  smallleftprocessed = cvCreateImage( cvSize(640,360), 8, 1 );
  smallrightprocessed = cvCreateImage( cvSize(640,360), 8, 1 );
  leftprocessed = cvCreateImage( cvSize(1920,540), 8, 1 );
  rightprocessed = cvCreateImage( cvSize(1920,540), 8, 1 );
  leftprocessed_temp = cvCreateImage( cvSize(1920,540), 8, 1 );
  rightprocessed_temp = cvCreateImage( cvSize(1920,540), 8, 1 );
  while ( key != 'q' )
  {
    leftframe = cvQueryFrame(capture);
    rightframe = cvQueryFrame(capture);

    cvCvtColor( leftframe, leftprocessed, CV_BGR2GRAY );
    cvCvtColor( rightframe, rightprocessed, CV_BGR2GRAY );
    
                                 
    cvEqualizeHist( leftprocessed, leftprocessed_temp );
    leftprocessed = leftprocessed_temp;
    cvEqualizeHist( rightprocessed, rightprocessed_temp );
    rightprocessed = rightprocessed_temp;
                                              

    //cvArr* leftCorners = new cvArr;
    //cvPreCornerDetect (leftframe, leftCorners);
    
    cvResize (leftframe, smallleft,CV_INTER_NN);
    cvResize (rightframe, smallright,CV_INTER_NN);
    
    cvResize (leftprocessed, smallleftprocessed,CV_INTER_NN);

    cvShowImage("Left Channel", smallleft);
    cvShowImage("Right Channel", smallright);
    cvShowImage("Left Processed", smallleftprocessed);
    key = cvWaitKey (50);
  }
  
  cvDestroyWindow("Left Channel");
  cvDestroyWindow("Right Channel");
  cvReleaseCapture (&capture);
  return 0;
}
