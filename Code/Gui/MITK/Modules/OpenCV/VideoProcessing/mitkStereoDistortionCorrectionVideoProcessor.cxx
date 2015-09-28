/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoDistortionCorrectionVideoProcessor.h"

namespace mitk
{

//-----------------------------------------------------------------------------
StereoDistortionCorrectionVideoProcessor::~StereoDistortionCorrectionVideoProcessor()
{
  cvReleaseMat(&m_IntrinsicLeft);
  cvReleaseMat(&m_DistortionLeft);
  cvReleaseImage(&m_MapXLeft);
  cvReleaseImage(&m_MapYLeft);

  cvReleaseMat(&m_IntrinsicRight);
  cvReleaseMat(&m_DistortionRight);
  cvReleaseImage(&m_MapXRight);
  cvReleaseImage(&m_MapYRight);
}


//-----------------------------------------------------------------------------
StereoDistortionCorrectionVideoProcessor::StereoDistortionCorrectionVideoProcessor(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer)
: StereoOneTimePointVideoProcessorTemplateMethod(writeInterleaved, capture, writer)
{
  throw std::logic_error("DistortionCorrectionVideoProcessor(CvCapture *capture, CvVideoWriter *writer) should not be called!");
}


//-----------------------------------------------------------------------------
StereoDistortionCorrectionVideoProcessor::StereoDistortionCorrectionVideoProcessor(
    const bool& writeInterleaved,
    const std::string& inputFile,
    const std::string& outputFile)
: StereoOneTimePointVideoProcessorTemplateMethod(writeInterleaved, inputFile, outputFile)
, m_IntrinsicLeft(NULL)
, m_DistortionLeft(NULL)
, m_MapXLeft(NULL)
, m_MapYLeft(NULL)
, m_IntrinsicRight(NULL)
, m_DistortionRight(NULL)
, m_MapXRight(NULL)
, m_MapYRight(NULL)
{
}


//-----------------------------------------------------------------------------
void StereoDistortionCorrectionVideoProcessor::SetMatrices(
    const CvMat& intrinsicLeft,
    const CvMat& distortionLeft,
    const CvMat& intrinsicRight,
    const CvMat& distortionRight
    )
{
  m_IntrinsicLeft = cvCloneMat(&intrinsicLeft);
  m_DistortionLeft = cvCloneMat(&distortionLeft);
  m_IntrinsicRight = cvCloneMat(&intrinsicRight);
  m_DistortionRight = cvCloneMat(&distortionRight);
}


//-----------------------------------------------------------------------------
void StereoDistortionCorrectionVideoProcessor::Initialize()
{
  // Mandatory. Important.
  StereoOneTimePointVideoProcessorTemplateMethod::Initialize();

  IplImage *image = this->GetCurrentImage();

  m_MapXLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicLeft, m_DistortionLeft, m_MapXLeft, m_MapYLeft);

  m_MapXRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicRight, m_DistortionRight, m_MapXRight, m_MapYRight);
}


//-----------------------------------------------------------------------------
void StereoDistortionCorrectionVideoProcessor::DoProcessing(
    const IplImage &leftInput, const IplImage &rightInput,
    IplImage &leftOutput, IplImage &rightOutput)
{
  cvRemap(&leftInput, &leftOutput, m_MapXLeft, m_MapYLeft);
  cvRemap(&rightInput, &rightOutput, m_MapXRight, m_MapYRight);
}

} // end namespace
