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

namespace mitk {

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
  cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
  CvCapture* capture = NULL;
  if (fileName.length() == 0)
  {
    capture = cvCreateCameraCapture(-1);
  }
  else
  {
    capture = cvCreateFileCapture(fileName.c_str());
  }
  IplImage *frame;
  while(1)
  {
    frame = cvQueryFrame(capture);
    if (!frame) break;
    cvShowImage("Example2", frame);
    char c = cvWaitKey(33);
    if (c == 27) break;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example2");
}

//-----------------------------------------------------------------------------
} // end namespace
