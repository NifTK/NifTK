/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkBaseVideoProcessor.h"

namespace mitk
{

//-----------------------------------------------------------------------------
BaseVideoProcessor::~BaseVideoProcessor()
{
  cvReleaseVideoWriter(&m_Writer);
  cvReleaseCapture(&m_Capture);
}


//-----------------------------------------------------------------------------
BaseVideoProcessor::BaseVideoProcessor(CvCapture *capture, CvVideoWriter *writer)
: m_GrabbedImage(NULL)
, m_Capture(capture)
, m_Writer(writer)
, m_InputFileName("")
, m_OutputFileName("")
{
  // For this constructor, we are assuming capture and writer are valid objects,
  // created outside this class and injected in, so bail out early if not true.

  if (m_Capture == NULL)
  {
    throw std::logic_error("Injected CvCapture is NULL.");
  }

  if (m_Writer == NULL)
  {
    throw std::logic_error("Injected CvVideoWriter is NULL.");
  }
}


//-----------------------------------------------------------------------------
BaseVideoProcessor::BaseVideoProcessor(const std::string& inputFile, const std::string& outputFile)
: m_GrabbedImage(NULL)
, m_Capture(NULL)
, m_Writer(NULL)
, m_InputFileName(inputFile)
, m_OutputFileName(outputFile)
{
  // For this constructor, we are assuming inputFile and outputFile are valid strings,
  // but at this stage, we are not checking if the files are valid, just that the strings are non-empty.

  if (inputFile == "")
  {
    throw std::logic_error("Injected input file name is empty.");
  }

  if (outputFile == "")
  {
    throw std::logic_error("Injected output file name is empty.");
  }
}


//-----------------------------------------------------------------------------
void BaseVideoProcessor::Initialize()
{
  if (m_Capture == NULL)
  {
    m_Capture = cvCreateFileCapture(m_InputFileName.c_str());
    if (m_Capture == NULL)
    {
      m_Capture = cvCreateCameraCapture(-1);
      if (m_Capture == NULL)
      {
        throw std::logic_error("Could not create capture device.");
      }
    }
  }

  // We call this, so that we have grabbed at least one frame, so we can do stuff like measure its size.
  this->GrabNewImage();

  if (m_Writer == NULL)
  {
    int inputCodec = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FOURCC);

    char codec1 = static_cast<char>(inputCodec & 0XFF);
    char codec2 = static_cast<char>((inputCodec & 0XFF00) >> 8);
    char codec3 = static_cast<char>((inputCodec & 0XFF0000) >> 16);
    char codec4 = static_cast<char>((inputCodec & 0XFF000000) >> 24);

    char inputCodecCode[] = {codec1, codec2, codec3, codec4, 0};

    int fps        = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FPS);
    int sizeX      = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_WIDTH);
    int sizeY      = (int)cvGetCaptureProperty(m_Capture,CV_CAP_PROP_FRAME_HEIGHT);

    CvSize outputSize = this->GetOutputImageSize();

    int outputFps = 25;
    int outputCodec = CV_FOURCC('D', 'I', 'V', 'X');

    codec1 = static_cast<char>(outputCodec & 0XFF);
    codec2 = static_cast<char>((outputCodec & 0XFF00) >> 8);
    codec3 = static_cast<char>((outputCodec & 0XFF0000) >> 16);
    codec4 = static_cast<char>((outputCodec & 0XFF000000) >> 24);

    char outputCodecCode[] = {codec1, codec2, codec3, codec4, 0};

    std::cout << "Input codec=" << inputCodec << ", " << inputCodecCode << ", fps=" << fps << ", size=(" << sizeX << ", " << sizeY << ")" << std::endl;
    std::cout << "Output codec=" << outputCodec << ", " << outputCodecCode << ", fps=" << outputFps << ", size=(" << sizeX << ", " << sizeY << ")" << std::endl;
    std::cout << "Output file=" << m_OutputFileName << std::endl;

    m_Writer = cvCreateVideoWriter(m_OutputFileName.c_str(), outputCodec, outputFps, outputSize);
    if (m_Writer == NULL)
    {
      throw std::logic_error("Could not create video writer.");
    }
    std::cout << "Created video file writer." << std::endl;
  }
}


//-----------------------------------------------------------------------------
IplImage* BaseVideoProcessor::GetCurrentImage() const
{
  if (m_GrabbedImage == NULL)
  {
    throw std::logic_error("The current image is NULL");
  }
  return m_GrabbedImage;
}


//-----------------------------------------------------------------------------
IplImage* BaseVideoProcessor::GrabNewImage()
{
  if (m_Capture == NULL)
  {
    throw std::logic_error("Failed to grab image as capture device is not initialised.");
  }

  // This will return NULL at end of file.
  m_GrabbedImage = cvQueryFrame(m_Capture);

  return m_GrabbedImage;
}


//-----------------------------------------------------------------------------
CvVideoWriter* BaseVideoProcessor::GetWriter() const
{
  if (m_Writer == NULL)
  {
    throw std::logic_error("The CvVideoWriter is NULL");
  }
  return m_Writer;
}

} // end namespace
