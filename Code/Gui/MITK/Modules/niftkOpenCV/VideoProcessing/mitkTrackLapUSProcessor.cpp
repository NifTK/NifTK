/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackLapUSProcessor.h"

namespace mitk
{

//-----------------------------------------------------------------------------
TrackLapUSProcessor::~TrackLapUSProcessor()
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
TrackLapUSProcessor::TrackLapUSProcessor(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer)
: StereoVideoProcessorTemplateMethod(writeInterleaved, capture, writer)
{
  throw std::logic_error("TrackLapUSProcessor(CvCapture *capture, CvVideoWriter *writer) should not be called!");
}


//-----------------------------------------------------------------------------
TrackLapUSProcessor::TrackLapUSProcessor(
    const bool& writeInterleaved,
    const std::string& inputFile,
    const std::string& outputFile)
: StereoVideoProcessorTemplateMethod(writeInterleaved, inputFile, outputFile)
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
void TrackLapUSProcessor::SetMatrices(
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
void TrackLapUSProcessor::Initialize()
{
  // Mandatory. Important.
  StereoVideoProcessorTemplateMethod::Initialize();

  IplImage *image = this->GetCurrentImage();

  m_MapXLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicLeft, m_DistortionLeft, m_MapXLeft, m_MapYLeft);

  m_MapXRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicRight, m_DistortionRight, m_MapXRight, m_MapYRight);
}


//-----------------------------------------------------------------------------
void TrackLapUSProcessor::DoProcessing(
    const IplImage &leftInput, const IplImage &rightInput,
    IplImage &leftOutput, IplImage &rightOutput)
{
  cvRemap(&leftInput, &leftOutput, m_MapXLeft, m_MapYLeft);
  cvRemap(&rightInput, &rightOutput, m_MapXRight, m_MapYRight);
}

} // end namespace
