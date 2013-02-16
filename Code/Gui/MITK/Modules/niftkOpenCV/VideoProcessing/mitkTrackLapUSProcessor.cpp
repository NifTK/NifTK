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
  cvReleaseImage(&m_GreyImageLeftT1);
  cvReleaseImage(&m_EigImageLeftT1);
  cvReleaseImage(&m_EigImageTmpLeftT1);
  cvReleaseImage(&m_PyramidLeftT1);
  cvReleaseImage(&m_GreyImageLeftT2);
  cvReleaseImage(&m_EigImageLeftT2);
  cvReleaseImage(&m_EigImageTmpLeftT2);
  cvReleaseImage(&m_PyramidLeftT2);

  cvReleaseMat(&m_IntrinsicRight);
  cvReleaseMat(&m_DistortionRight);
  cvReleaseImage(&m_MapXRight);
  cvReleaseImage(&m_MapYRight);
  cvReleaseImage(&m_GreyImageRightT1);
  cvReleaseImage(&m_EigImageRightT1);
  cvReleaseImage(&m_EigImageTmpRightT1);
  cvReleaseImage(&m_PyramidRightT1);
  cvReleaseImage(&m_GreyImageRightT2);
  cvReleaseImage(&m_EigImageRightT2);
  cvReleaseImage(&m_EigImageTmpRightT2);
  cvReleaseImage(&m_PyramidRightT2);
}


//-----------------------------------------------------------------------------
TrackLapUSProcessor::TrackLapUSProcessor(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer)
: StereoTwoTimePointVideoProcessorTemplateMethod(writeInterleaved, capture, writer)
{
  throw std::logic_error("TrackLapUSProcessor(CvCapture *capture, CvVideoWriter *writer) should not be called!");
}


//-----------------------------------------------------------------------------
TrackLapUSProcessor::TrackLapUSProcessor(
    const bool& writeInterleaved,
    const std::string& inputFile,
    const std::string& outputFile)
: StereoTwoTimePointVideoProcessorTemplateMethod(writeInterleaved, inputFile, outputFile)
, m_IntrinsicLeft(NULL)
, m_DistortionLeft(NULL)
, m_MapXLeft(NULL)
, m_MapYLeft(NULL)
, m_GreyImageLeftT1(NULL)
, m_EigImageLeftT1(NULL)
, m_EigImageTmpLeftT1(NULL)
, m_PyramidLeftT1(NULL)
, m_GreyImageLeftT2(NULL)
, m_EigImageLeftT2(NULL)
, m_EigImageTmpLeftT2(NULL)
, m_PyramidLeftT2(NULL)
, m_IntrinsicRight(NULL)
, m_DistortionRight(NULL)
, m_MapXRight(NULL)
, m_MapYRight(NULL)
, m_GreyImageRightT1(NULL)
, m_EigImageRightT1(NULL)
, m_EigImageTmpRightT1(NULL)
, m_PyramidRightT1(NULL)
, m_GreyImageRightT2(NULL)
, m_EigImageRightT2(NULL)
, m_EigImageTmpRightT2(NULL)
, m_PyramidRightT2(NULL)
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
  StereoTwoTimePointVideoProcessorTemplateMethod::Initialize();

  IplImage *image = this->GetCurrentImage();

  m_MapXLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYLeft = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicLeft, m_DistortionLeft, m_MapXLeft, m_MapYLeft);

  m_GreyImageLeftT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
  m_EigImageLeftT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_EigImageTmpLeftT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_GreyImageLeftT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
  m_EigImageLeftT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_EigImageTmpLeftT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

  m_MapXRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_MapYRight = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  cvInitUndistortMap(m_IntrinsicRight, m_DistortionRight, m_MapXRight, m_MapYRight);

  m_GreyImageRightT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
  m_EigImageRightT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_EigImageTmpRightT1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_GreyImageRightT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
  m_EigImageRightT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
  m_EigImageTmpRightT2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

  CvSize pyramidSize = cvSize(image->width+8, image->height/3);
  m_PyramidLeftT1 = cvCreateImage(pyramidSize, IPL_DEPTH_32F, 1);
  m_PyramidLeftT2 = cvCreateImage(pyramidSize, IPL_DEPTH_32F, 1);
  m_PyramidRightT1 = cvCreateImage(pyramidSize, IPL_DEPTH_32F, 1);
  m_PyramidRightT2 = cvCreateImage(pyramidSize, IPL_DEPTH_32F, 1);

}


//-----------------------------------------------------------------------------
void TrackLapUSProcessor::DoProcessing(
    const IplImage &leftInputT1,
    const IplImage &rightInputT1,
    const IplImage &leftInputT2,
    const IplImage &rightInputT2,
    IplImage &leftOutput,
    IplImage &rightOutput)
{
  // Distortion Correct T1
  cvRemap(&leftInputT1, &leftOutput, m_MapXLeft, m_MapYLeft);
  cvRemap(&rightInputT1, &rightOutput, m_MapXRight, m_MapYRight);

  // Convert to Grey T1
  cvCvtColor(&leftOutput, m_GreyImageLeftT1, CV_RGB2GRAY);
  cvCvtColor(&rightOutput, m_GreyImageRightT1, CV_RGB2GRAY);

  // Distortion Correct T2
  cvRemap(&leftInputT2, &leftOutput, m_MapXLeft, m_MapYLeft);
  cvRemap(&rightInputT2, &rightOutput, m_MapXRight, m_MapYRight);

  // Convert to Grey T2
  cvCvtColor(&leftOutput, m_GreyImageLeftT2, CV_RGB2GRAY);
  cvCvtColor(&rightOutput, m_GreyImageRightT2, CV_RGB2GRAY);

  //******************
  // Hunt for corners.
  // *****************

  int windowSize = 10;
  int MAX_CORNERS = 500;

  CvPoint2D32f* cornersLeftT1 = new CvPoint2D32f[MAX_CORNERS];
  CvPoint2D32f* cornersLeftT2 = new CvPoint2D32f[MAX_CORNERS];
  CvPoint2D32f* cornersRightT1 = new CvPoint2D32f[MAX_CORNERS];
  CvPoint2D32f* cornersRightT2 = new CvPoint2D32f[MAX_CORNERS];
  int cornerCountLeft = MAX_CORNERS;
  int cornerCountRight = MAX_CORNERS;
  char featuresFoundLeft[MAX_CORNERS];
  float featureErrorsLeft[MAX_CORNERS];
  char featuresFoundRight[MAX_CORNERS];
  float featureErrorsRight[MAX_CORNERS];

  // Lucas Kanade sparse optical flow
  cvGoodFeaturesToTrack(m_GreyImageLeftT1, m_EigImageLeftT1, m_EigImageTmpLeftT1, cornersLeftT1, &cornerCountLeft, 0.1, 3.0, 0, 5, 0, 0.04);
  cvCalcOpticalFlowPyrLK(m_GreyImageLeftT1, m_GreyImageLeftT2, m_PyramidLeftT1, m_PyramidLeftT2, cornersLeftT1, cornersLeftT2, cornerCountLeft, cvSize(windowSize, windowSize), 5, featuresFoundLeft, featureErrorsLeft, cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, -3), 0);
  for (int i = 0; i < cornerCountLeft; i++)
  {
    if (featuresFoundLeft[i] == 0 || featureErrorsLeft[i] > 550)
    {
      continue;
    }
    CvPoint p0 = cvPoint(cvRound(cornersLeftT1[i].x), cvRound(cornersLeftT1[i].y));
    CvPoint p1 = cvPoint(cvRound(cornersLeftT2[i].x), cvRound(cornersLeftT2[i].y));
    cvLine (&leftOutput, p0, p1, CV_RGB(255, 0, 0), 2);
  }

  // Lucas Kanade sparse optical flow
  cvGoodFeaturesToTrack(m_GreyImageRightT1, m_EigImageRightT1, m_EigImageTmpRightT1, cornersRightT1, &cornerCountRight, 0.1, 3.0, 0, 5, 0, 0.04);
  cvCalcOpticalFlowPyrLK(m_GreyImageRightT1, m_GreyImageRightT2, m_PyramidRightT1, m_PyramidRightT2, cornersRightT1, cornersRightT2, cornerCountRight, cvSize(windowSize, windowSize), 5, featuresFoundRight, featureErrorsRight, cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, -3), 0);
  for (int i = 0; i < cornerCountRight; i++)
  {
    if (featuresFoundRight[i] == 0 || featureErrorsRight[i] > 550)
    {
      continue;
    }
    CvPoint p0 = cvPoint(cvRound(cornersRightT1[i].x), cvRound(cornersRightT1[i].y));
    CvPoint p1 = cvPoint(cvRound(cornersRightT2[i].x), cvRound(cornersRightT2[i].y));
    cvLine (&rightOutput, p0, p1, CV_RGB(255, 0, 0), 2);
  }

}

} // end namespace
