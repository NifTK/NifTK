/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoOneTimePointVideoProcessorTemplateMethod.h"

namespace mitk
{

//-----------------------------------------------------------------------------
StereoOneTimePointVideoProcessorTemplateMethod::~StereoOneTimePointVideoProcessorTemplateMethod()
{
  cvReleaseImage(&m_LeftInput);
  cvReleaseImage(&m_RightInput);
  cvReleaseImage(&m_LeftOutput);
  cvReleaseImage(&m_RightOutput);
}


//-----------------------------------------------------------------------------
StereoOneTimePointVideoProcessorTemplateMethod::StereoOneTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer)
: StereoVideoProcessorTemplateMethod(writeInterleaved, capture, writer)
, m_LeftInput(NULL)
, m_RightInput(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
{
}


//-----------------------------------------------------------------------------
StereoOneTimePointVideoProcessorTemplateMethod::StereoOneTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile)
: StereoVideoProcessorTemplateMethod(writeInterleaved, inputFile, outputFile)
, m_LeftInput(NULL)
, m_RightInput(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
{
}


//-----------------------------------------------------------------------------
void StereoOneTimePointVideoProcessorTemplateMethod::Initialize()
{
  // Mandatory. Important.
  StereoVideoProcessorTemplateMethod::Initialize();

  // Get base class image.
  IplImage *image = this->GetCurrentImage();

  // Allocate new buffers as appropriate.
  m_LeftInput   = cvCloneImage(image);
  m_RightInput  = cvCloneImage(image);
  m_LeftOutput  = cvCloneImage(image);
  m_RightOutput = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
void StereoOneTimePointVideoProcessorTemplateMethod::Run()
{
  IplImage *image = NULL;

  while((image = this->GrabNewImage()) != NULL)
  {
    cvCopy(image, m_LeftInput);

    image = this->GrabNewImage();
    if (image == NULL)
    {
      break;
    }

    cvCopy(image, m_RightInput);

    this->DoProcessing(
        *m_LeftInput,
        *m_RightInput,
        *m_LeftOutput,
        *m_RightOutput
        );

    this->WriteOutput(*m_LeftOutput, *m_RightOutput);
  }
}

} // end namespace
