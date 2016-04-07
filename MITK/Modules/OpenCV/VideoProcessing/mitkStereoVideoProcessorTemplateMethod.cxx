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
}


//-----------------------------------------------------------------------------
StereoVideoProcessorTemplateMethod::StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer
    )
: BaseVideoProcessor(capture, writer)
, m_WriteInterleaved(writeInterleaved)
, m_OutputImage(NULL)
, m_FrameCount(0)
{
}


//-----------------------------------------------------------------------------
StereoVideoProcessorTemplateMethod::StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile)
: BaseVideoProcessor(inputFile, outputFile)
, m_WriteInterleaved(writeInterleaved)
, m_OutputImage(NULL)
, m_FrameCount(0)
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

  // Instead of calling this->GetCurrentImage(), we explicitly force a new grab, so
  // that if we are reading from file, the base class method has read 1 frame, and here we read 1 frame.
  // So we have dropped 2 frames, but at least we are in sequence. (i.e. we wont mix up odd and even frames).
  IplImage *image = this->GrabNewImage();

  // Now we allocate the size of the output.
  CvSize outputSize = this->GetOutputImageSize();
  m_OutputImage = cvCreateImage(outputSize, image->depth, image->nChannels);

  // Reset this.
  m_FrameCount = 0;
}


//-----------------------------------------------------------------------------
void StereoVideoProcessorTemplateMethod::WriteOutput(IplImage &leftOutput, IplImage &rightOutput)
{
  CvVideoWriter* writer = this->GetWriter();

  if (m_WriteInterleaved)
  {
    cvCopy(&leftOutput, m_OutputImage);
    cvWriteFrame(writer, m_OutputImage);
    std::cout << "Written output left frame " << ++m_FrameCount << std::endl;

    cvCopy(&rightOutput, m_OutputImage);
    cvWriteFrame(writer, m_OutputImage);
    std::cout << "Written output right frame " << ++m_FrameCount << std::endl;
  }
  else
  {
    // Copy left and right, side by side
    cvSetImageROI(&leftOutput, cvRect(0,0,leftOutput.width, leftOutput.height));
    cvSetImageROI(m_OutputImage, cvRect(0,0,leftOutput.width, leftOutput.height));
    cvCopy(&leftOutput, m_OutputImage);

    cvSetImageROI(&rightOutput, cvRect(0,0,rightOutput.width, rightOutput.height));
    cvSetImageROI(m_OutputImage, cvRect(rightOutput.width,0,rightOutput.width, rightOutput.height));
    cvCopy(&rightOutput, m_OutputImage);

    cvWriteFrame(writer, m_OutputImage);
    std::cout << "Written output side by-side frame " << ++m_FrameCount << std::endl;
  }
}


} // end namespace
