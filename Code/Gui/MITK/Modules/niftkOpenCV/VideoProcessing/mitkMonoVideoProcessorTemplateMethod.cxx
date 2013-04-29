/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMonoVideoProcessorTemplateMethod.h"

namespace mitk
{

//-----------------------------------------------------------------------------
MonoVideoProcessorTemplateMethod::~MonoVideoProcessorTemplateMethod()
{
  cvReleaseImage(&m_OutputImage);
}


//-----------------------------------------------------------------------------
MonoVideoProcessorTemplateMethod::MonoVideoProcessorTemplateMethod(CvCapture *capture, CvVideoWriter *writer)
: BaseVideoProcessor(capture, writer)
, m_OutputImage(NULL)
{
}


//-----------------------------------------------------------------------------
MonoVideoProcessorTemplateMethod::MonoVideoProcessorTemplateMethod(const std::string& inputFile, const std::string& outputFile)
: BaseVideoProcessor(inputFile, outputFile)
, m_OutputImage(NULL)
{
}


//-----------------------------------------------------------------------------
CvSize MonoVideoProcessorTemplateMethod::GetOutputImageSize()
{
  IplImage *image = this->GetCurrentImage();
  return cvSize(image->width, image->height);
}


//-----------------------------------------------------------------------------
void MonoVideoProcessorTemplateMethod::Initialize()
{
  // Mandatory. Important.
  BaseVideoProcessor::Initialize();

  // This class requires an extra buffer to store the output image.
  IplImage *image = this->GetCurrentImage();
  m_OutputImage = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
void MonoVideoProcessorTemplateMethod::Run()
{
  int frame = 0;
  IplImage *image = NULL;
  CvVideoWriter* writer = this->GetWriter();

  while((image = this->GrabNewImage()) != NULL)
  {
    this->DoProcessing(*image, *m_OutputImage);
    cvWriteFrame(writer, m_OutputImage);
    std::cout << "Written mono output frame " << ++frame << std::endl;
  }
}

} // end namespace
