/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoProcessorTemplateMethod.h"

namespace mitk
{

//-----------------------------------------------------------------------------
VideoProcessorTemplateMethod::~VideoProcessorTemplateMethod()
{
  if (m_Capture != NULL)
  {
    cvReleaseCapture(&m_Capture);
  }
  if (m_Writer != NULL)
  {
    cvReleaseVideoWriter(&m_Writer);
  }
}


//-----------------------------------------------------------------------------
VideoProcessorTemplateMethod::VideoProcessorTemplateMethod(
    CvCapture *capture, CvVideoWriter *writer
    )
: m_Capture(capture)
, m_Writer(writer)
{
}


//-----------------------------------------------------------------------------
VideoProcessorTemplateMethod::VideoProcessorTemplateMethod(
    const std::string& inputFile, const std::string& outputFile)
{
  m_Capture = cvCreateFileCapture(inputFile.c_str());
  if (m_Capture == NULL)
  {
    throw std::logic_error("Could not create video file reader.");
  }

  IplImage* image = cvQueryFrame(m_Capture);

  if (outputFile.size() > 0 && image != NULL)
  {
    int ex     = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FOURCC);
    int fps    = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FPS);
    int sizeX  = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_WIDTH);
    int sizeY  = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_HEIGHT);

    CvSize size = cvSize(sizeX, sizeY);
    char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0};

    std::cout << "Input codec=" << ex << ", " << EXT << ", fps=" << fps << ", size=(" << sizeX << ", " << sizeY << ")" << std::endl;
    std::cout << "Output file=" << outputFile << std::endl;

    m_Writer = cvCreateVideoWriter(outputFile.c_str(), CV_FOURCC('D','I','V','X'), fps, size);
    if (m_Writer == NULL)
    {
      throw std::logic_error("Could not open video output");
    }
  }
}


//-----------------------------------------------------------------------------
void VideoProcessorTemplateMethod::Run()
{
  if (m_Capture == NULL)
  {
    // Last resort, try to initialise an on-board camera.
    m_Capture = cvCreateCameraCapture(-1);
  }
  if (m_Capture == NULL)
  {
    throw std::logic_error("No file-name specified, and could not create video source");
  }
  if (m_Writer == NULL)
  {
    throw std::logic_error("No output writer specified");
  }

  IplImage *input = cvQueryFrame(m_Capture);
  IplImage *output = cvCloneImage(input);

  while((input = cvQueryFrame(m_Capture)) != NULL)
  {
    this->DoProcessing(*input, *output);
    cvWriteFrame(m_Writer, output);
  }
}


//-----------------------------------------------------------------------------
IplImage* VideoProcessorTemplateMethod::GetImage()
{
  if (m_Capture == NULL)
  {
    throw std::logic_error("You must initialise the capture device before calling GetImage()");
  }
  return cvQueryFrame(m_Capture);
}

} // end namespace
