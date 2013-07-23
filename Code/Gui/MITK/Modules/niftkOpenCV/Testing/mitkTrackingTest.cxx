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
#include <opencv2/imgproc/imgproc.hpp>


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
  argv ++; 
  argc --; 
  
  //hough parameters
  double rho = 10.0;
  double theta = 0.1;
  int threshold = 10;
  int linelength = 10;
  int linegap = 2;

  //canny parameters
  int lowThreshold = 100;
  int highThreshold = 100;
  int kernel = 3;
  while ( argc > 1 )
  {
    bool ok = false; 
    if ( ( ok == false ) && (strcmp ( argv[1] , "-hough" ) == 0) ) 
    {
      rho = atof(argv[2]); //10.0;
      theta = atof (argv[3]); //0.1;
      threshold = atoi (argv[4]); //10;
      linelength = atoi (argv[5]); // 10;
      linegap = atoi (argv[6]) ; //2;
      argv += 6;
      argc -= 6;
      ok =true;
    }
    if (( ok ==false ) && strcmp ( argv[1] , "-canny" ) == 0 ) 
    {
      lowThreshold = atoi(argv[2]);
      highThreshold = atoi(argv[3]);
      kernel = atoi(argv[4]);
      argv += 4;
      argc -= 4;
      ok =true;
    }
    if ( ok == false ) 
    {
      MITK_WARN << "Bad parameters.";
      exit (1) ;
    }
  }

      
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
  IplImage *framegrab;
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
  rightframe = cvCreateImage( cvSize(1920,540), 8, 3 );
  leftframe = cvCreateImage( cvSize(1920,540), 8, 3 );

  
  int framecount=0;
  while ( key != 'q' )
  {
    framegrab =  cvQueryFrame(capture);
    cvCopyImage(framegrab,leftframe);
    framegrab =  cvQueryFrame(capture);
    cvCopyImage(framegrab,rightframe);
    MITK_INFO << leftframe << " " << rightframe;

    cvCvtColor( leftframe, leftprocessed, CV_BGR2GRAY );
    cvCvtColor( rightframe, rightprocessed, CV_BGR2GRAY );
    
                                 
    cvEqualizeHist( leftprocessed, leftprocessed_temp );
    leftprocessed = leftprocessed_temp;
    cvEqualizeHist( rightprocessed, rightprocessed_temp );
    rightprocessed = rightprocessed_temp;
                                              
    IplImage *temp;
    temp=cvCreateImage(cvSize(1920,540),32, 1);
    IplImage *smalltemp;
    smalltemp=cvCreateImage(cvSize(640,360),32, 1);
    
    cv::Mat leftprocessedMat(leftprocessed);
    cv::Mat leftprocessed_tempMat(leftprocessed_temp);
    cv::blur ( leftprocessedMat, leftprocessed_tempMat, cv::Size(3,3));
    leftprocessed_tempMat = leftprocessedMat;
    cv::Canny ( leftprocessedMat, leftprocessed_tempMat, lowThreshold , highThreshold , kernel);
    
    cv::vector<cv::Vec4i> lines2;
    cv::HoughLinesP (leftprocessed_tempMat, lines2,rho,theta, threshold, linelength , linegap);
    MITK_INFO << "Frame " << framecount++ << " found " << lines2.size() << " lines"; 
    for ( unsigned int i = 0 ; i < lines2.size() ; i ++ ) 
    {
      cv::Vec4i l = lines2[i];
      cv::Mat TL(leftframe);
      cv::Mat TR (rightframe);
      cv::line(TL , cvPoint(l[0], l[1]),
          cvPoint(l[2],l[3]), cvScalar(255,0,0));
      cv::line(TR , cvPoint(l[0], l[1]),
          cvPoint(l[2],l[3]), cvScalar(0,255,0));
    }
    cvResize (leftframe, smallleft,CV_INTER_NN);
    cvResize (rightframe, smallright,CV_INTER_NN);
    
    cvResize (leftprocessed_temp, smallleftprocessed,CV_INTER_NN);
   // cvResize (temp, smalltemp,CV_INTER_NN);

    cvShowImage("Left Channel", smallleft);
    cvShowImage("Right Channel", smallright);
    cvShowImage("Left Processed", smallleftprocessed);
  //  cvShowImage("Left Processed", smalltemp);
    key = cvWaitKey (1);
 //   key = 'q';
  }
  
  cvDestroyWindow("Left Channel");
  cvDestroyWindow("Right Channel");
  cvReleaseCapture (&capture);
  return 0;
}
