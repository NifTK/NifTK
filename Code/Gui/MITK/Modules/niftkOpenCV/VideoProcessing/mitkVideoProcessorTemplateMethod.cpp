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

  m_GrabbedImage = cvQueryFrame(m_Capture);
  if (m_GrabbedImage == NULL)
  {
    throw std::logic_error("Failed to grab image from capture device.");
  }

  if (outputFile.size() > 0)
  {
    int inputCodec = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FOURCC);
    char inputCodecCode[] = {inputCodec & 0XFF , (inputCodec & 0XFF00) >> 8,(inputCodec & 0XFF0000) >> 16,(inputCodec & 0XFF000000) >> 24, 0};

    int fps        = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FPS);
    int sizeX      = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_WIDTH);
    int sizeY      = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_HEIGHT);

    CvSize size = cvSize(sizeX, sizeY);

    int outputFps = 25;
    int outputCodec = CV_FOURCC('D', 'I', 'V', 'X');
    char outputCodecCode[] = {outputCodec & 0XFF , (outputCodec & 0XFF00) >> 8,(outputCodec & 0XFF0000) >> 16,(outputCodec & 0XFF000000) >> 24, 0};

    std::cout << "Input codec=" << inputCodec << ", " << inputCodecCode << ", fps=" << fps << ", size=(" << sizeX << ", " << sizeY << ")" << std::endl;
    std::cout << "Output codec=" << outputCodec << ", " << outputCodecCode << ", fps=" << outputFps << ", size=(" << sizeX << ", " << sizeY << ")" << std::endl;
    std::cout << "Output file=" << outputFile << std::endl;

    m_Writer = cvCreateVideoWriter(outputFile.c_str(), outputCodec, outputFps, size);
    if (m_Writer == NULL)
    {
      throw std::logic_error("Could not open video output");
    }
    std::cout << "Created writer" << std::endl;
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

  m_GrabbedImage = cvQueryFrame(m_Capture);
  IplImage *output = cvCloneImage(m_GrabbedImage);

  int frame = 0;
  while((m_GrabbedImage = cvQueryFrame(m_Capture)) != NULL)
  {
    this->DoProcessing(*m_GrabbedImage, *output);
    cvWriteFrame(m_Writer, output);
    std::cout << "Processed frame " << frame++ << std::endl;
  }

  cvReleaseImage(&output);
}


//-----------------------------------------------------------------------------
IplImage* VideoProcessorTemplateMethod::GetImage()
{
  return m_GrabbedImage;
}

} // end namespace
