/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDistortionCorrectionVideoProcessor.h"

namespace mitk
{

//-----------------------------------------------------------------------------
DistortionCorrectionVideoProcessor::~DistortionCorrectionVideoProcessor()
{
  if (m_MapX != NULL)
  {
    cvReleaseImage(&m_MapX);
  }
  if (m_MapY != NULL)
  {
    cvReleaseImage(&m_MapY);
  }
  cvReleaseMat(&m_Intrinsic);
  cvReleaseMat(&m_Distortion);
}


//-----------------------------------------------------------------------------
DistortionCorrectionVideoProcessor::DistortionCorrectionVideoProcessor(
    CvCapture *capture, CvVideoWriter *writer
    )
: VideoProcessorTemplateMethod(capture, writer)
{
  throw std::logic_error("DistortionCorrectionVideoProcessor(CvCapture *capture, CvVideoWriter *writer) should not be called!");
}


//-----------------------------------------------------------------------------
DistortionCorrectionVideoProcessor::DistortionCorrectionVideoProcessor(
    const std::string& inputFile,
    const std::string& outputFile,
    const CvMat& intrinsicParams,
    const CvMat& distortionParams
    )
: VideoProcessorTemplateMethod(inputFile, outputFile)
{
  m_Intrinsic = cvCloneMat(&intrinsicParams);
  m_Distortion = cvCloneMat(&distortionParams);

  IplImage *image = this->GetImage();
  if (image != NULL)
  {
    m_MapX = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    m_MapY = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvInitUndistortMap(m_Intrinsic, m_Distortion, m_MapX, m_MapY);
  }
}


//-----------------------------------------------------------------------------
void DistortionCorrectionVideoProcessor::DoProcessing(const IplImage &input, IplImage &output)
{
  cvRemap(&input, &output, m_MapX, m_MapY);
}

} // end namespace
