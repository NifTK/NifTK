/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKTRACKLAPUSPROCESSOR_H
#define MITKTRACKLAPUSPROCESSOR_H

#include "niftkOpenCVExports.h"
#include "mitkStereoTwoTimePointVideoProcessorTemplateMethod.h"

/**
 * class TrackLapUSProcessor
 * \brief Derived from StereoTwoTimePointVideoProcessorTemplateMethod to track Laparoscopic Ultrasound in interleaved (flicker) stereo stream.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT TrackLapUSProcessor : public StereoTwoTimePointVideoProcessorTemplateMethod
{

public:

  mitkClassMacro(TrackLapUSProcessor, StereoTwoTimePointVideoProcessorTemplateMethod);
  mitkNewMacro3Param(TrackLapUSProcessor, const bool&, const std::string&, const std::string&);

  /**
   * \brief Call before TrackLapUSProcessor::Initialize().
   */
  void SetMatrices(
      const CvMat& intrinsicLeft,
      const CvMat& distortionLeft,
      const CvMat& intrinsicRight,
      const CvMat& distortionRight
      );

  /**
   * \see BaseVideoProcessor::Initialize()
   */
  virtual void Initialize();

protected:

  ~TrackLapUSProcessor();
  TrackLapUSProcessor(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer);
  TrackLapUSProcessor(const bool& writeInterleaved, const std::string&, const std::string&);

  TrackLapUSProcessor(const TrackLapUSProcessor&); // Purposefully not implemented.
  TrackLapUSProcessor& operator=(const TrackLapUSProcessor&); // Purposefully not implemented.

  virtual void DoProcessing(
      const IplImage &leftInputT1,
      const IplImage &rightInputT1,
      const IplImage &leftInputT2,
      const IplImage &rightInputT2,
      IplImage &leftOutput,
      IplImage &rightOutput);

private:

  CvMat *m_IntrinsicLeft;
  CvMat *m_DistortionLeft;
  IplImage *m_MapXLeft;
  IplImage *m_MapYLeft;
  IplImage *m_GreyImageLeftT1;
  IplImage *m_EigImageLeftT1;
  IplImage *m_EigImageTmpLeftT1;
  IplImage *m_PyramidLeftT1;
  IplImage *m_GreyImageLeftT2;
  IplImage *m_EigImageLeftT2;
  IplImage *m_EigImageTmpLeftT2;
  IplImage *m_PyramidLeftT2;

  CvMat *m_IntrinsicRight;
  CvMat *m_DistortionRight;
  IplImage *m_MapXRight;
  IplImage *m_MapYRight;
  IplImage *m_GreyImageRightT1;
  IplImage *m_EigImageRightT1;
  IplImage *m_EigImageTmpRightT1;
  IplImage *m_PyramidRightT1;
  IplImage *m_GreyImageRightT2;
  IplImage *m_EigImageRightT2;
  IplImage *m_EigImageTmpRightT2;
  IplImage *m_PyramidRightT2;
}; // end class

} // end namespace

#endif // MITKSTEREODISTORTIONCORRECTIONVIDEOPROCESSOR_H
