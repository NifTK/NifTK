/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoTwoTimePointVideoProcessorTemplateMethod.h"

namespace mitk
{

//-----------------------------------------------------------------------------
StereoTwoTimePointVideoProcessorTemplateMethod::~StereoTwoTimePointVideoProcessorTemplateMethod()
{
  cvReleaseImage(&m_LeftT1);
  cvReleaseImage(&m_RightT1);
  cvReleaseImage(&m_LeftT2);
  cvReleaseImage(&m_RightT2);
  cvReleaseImage(&m_LeftOutput);
  cvReleaseImage(&m_RightOutput);
}


//-----------------------------------------------------------------------------
StereoTwoTimePointVideoProcessorTemplateMethod::StereoTwoTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer)
: StereoVideoProcessorTemplateMethod(writeInterleaved, capture, writer)
, m_LeftT1(NULL)
, m_RightT1(NULL)
, m_LeftT2(NULL)
, m_RightT2(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
{
}


//-----------------------------------------------------------------------------
StereoTwoTimePointVideoProcessorTemplateMethod::StereoTwoTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile)
: StereoVideoProcessorTemplateMethod(writeInterleaved, inputFile, outputFile)
, m_LeftT1(NULL)
, m_RightT1(NULL)
, m_LeftT2(NULL)
, m_RightT2(NULL)
, m_LeftOutput(NULL)
, m_RightOutput(NULL)
{
}


//-----------------------------------------------------------------------------
void StereoTwoTimePointVideoProcessorTemplateMethod::Initialize()
{
  // Mandatory. Important.
  StereoVideoProcessorTemplateMethod::Initialize();

  // Get base class image.
  IplImage *image = this->GetCurrentImage();

  // Allocate new buffers as appropriate.
  m_LeftT1   = cvCloneImage(image);
  m_RightT1  = cvCloneImage(image);
  m_LeftT2   = cvCloneImage(image);
  m_RightT2  = cvCloneImage(image);
  m_LeftOutput  = cvCloneImage(image);
  m_RightOutput = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
void StereoTwoTimePointVideoProcessorTemplateMethod::Run()
{
  IplImage *image = NULL;

  image = this->GrabNewImage();
  if (image != NULL)
  {
    cvCopy(image, m_LeftT1);
  }
  else
  {
    return;
  }

  image = this->GrabNewImage();
  if (image != NULL)
  {
    cvCopy(image, m_RightT1);
  }
  else
  {
    return;
  }

  while(image != NULL)
  {
    image = this->GrabNewImage();
    if (image == NULL)
    {
      break;
    }
    cvCopy(image, m_LeftT2);

    image = this->GrabNewImage();
    if (image == NULL)
    {
      break;
    }
    cvCopy(image, m_RightT2);

    this->DoProcessing(
        *m_LeftT1,
        *m_RightT1,
        *m_LeftT2,
        *m_RightT2,
        *m_LeftOutput,
        *m_RightOutput
        );

    this->WriteOutput(*m_LeftOutput, *m_RightOutput);

    cvCopy(m_LeftT2, m_LeftT1);
    cvCopy(m_RightT2, m_RightT1);
  }
}

} // end namespace
