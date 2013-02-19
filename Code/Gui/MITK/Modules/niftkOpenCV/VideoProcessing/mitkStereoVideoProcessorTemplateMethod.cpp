/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoVideoProcessorTemplateMethod.h"

namespace mitk
{

//-----------------------------------------------------------------------------
StereoVideoProcessorTemplateMethod::~StereoVideoProcessorTemplateMethod()
{
  cvReleaseImage(&m_LeftInput);
  cvReleaseImage(&m_RightInput);
  cvReleaseImage(&m_LeftOutput);
  cvReleaseImage(&m_RightOutput);
  cvReleaseImage(&m_OutputImage);
}


//-----------------------------------------------------------------------------
StereoVideoProcessorTemplateMethod::StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer
    )
: BaseVideoProcessor(capture, writer)
, m_LeftInput(NULL)
, m_RightInput(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
, m_OutputImage(NULL)
, m_WriteInterleaved(writeInterleaved)
{
}


//-----------------------------------------------------------------------------
StereoVideoProcessorTemplateMethod::StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile)
: BaseVideoProcessor(inputFile, outputFile)
, m_LeftInput(NULL)
, m_RightInput(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
, m_OutputImage(NULL)
, m_WriteInterleaved(writeInterleaved)
{
}


//-----------------------------------------------------------------------------
CvSize StereoVideoProcessorTemplateMethod::GetOutputImageSize()
{
  IplImage *image = this->GetCurrentImage();
  if (m_WriteInterleaved)
  {
    return cvSize(image->width, image->height);
  }
  else
  {
    return cvSize(image->width*2, image->height);
  }
}


//-----------------------------------------------------------------------------
void StereoVideoProcessorTemplateMethod::Initialize()
{
  // Mandatory. Important.
  BaseVideoProcessor::Initialize();

  // This provides a reference size and type for the input image.
  // Instead of calling this->GetCurrentImage(), we explicitly force a new grab, so
  // that if we are reading from file, the base class method has read 1 frame, and here we read 1 frame.
  // So we have dropped 2 frames, but at least we are in sequence. (i.e. we wont mix up odd and even frames).
  IplImage *image = this->GrabNewImage();

  // Allocate new buffers as appropriate.
  m_LeftInput   = cvCloneImage(image);
  m_RightInput  = cvCloneImage(image);
  m_LeftOutput  = cvCloneImage(image);
  m_RightOutput = cvCloneImage(image);

  CvSize outputSize = this->GetOutputImageSize();
  m_OutputImage = cvCreateImage(outputSize, image->depth, image->nChannels);
}


//-----------------------------------------------------------------------------
void StereoVideoProcessorTemplateMethod::Run()
{
  int frame = 0;
  IplImage *image = NULL;
  CvVideoWriter* writer = this->GetWriter();

  while((image = this->GrabNewImage()) != NULL)
  {
    cvCopy(image, m_LeftInput);
    image = this->GrabNewImage();

    if (image == NULL)
    {
      // Stereo pair failed
      break;
    }

    cvCopy(image, m_RightInput);

    this->DoProcessing(*m_LeftInput, *m_RightInput, *m_LeftOutput, *m_RightOutput);

    if (m_WriteInterleaved)
    {
      cvCopy(m_LeftOutput, m_OutputImage);
      cvWriteFrame(writer, m_OutputImage);
      std::cout << "Written output left frame " << ++frame << std::endl;

      cvCopy(m_RightOutput, m_OutputImage);
      cvWriteFrame(writer, m_OutputImage);
      std::cout << "Written output right frame " << ++frame << std::endl;
    }
    else
    {
      // Copy left and right, side by side
      cvSetImageROI(m_LeftOutput, cvRect(0,0,m_LeftOutput->width, m_LeftOutput->height));
      cvSetImageROI(m_OutputImage, cvRect(0,0,m_LeftOutput->width, m_LeftOutput->height));
      cvCopy(m_LeftOutput, m_OutputImage);

      cvSetImageROI(m_RightOutput, cvRect(0,0,m_RightOutput->width, m_RightOutput->height));
      cvSetImageROI(m_OutputImage, cvRect(m_LeftOutput->width,0,m_RightOutput->width, m_RightOutput->height));
      cvCopy(m_RightOutput, m_OutputImage);

      cvWriteFrame(writer, m_OutputImage);
      std::cout << "Written output side by-side frame " << ++frame << std::endl;
    }
  }
}

} // end namespace
