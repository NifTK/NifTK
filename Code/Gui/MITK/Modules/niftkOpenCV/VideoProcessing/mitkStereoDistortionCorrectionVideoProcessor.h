/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKSTEREODISTORTIONCORRECTIONVIDEOPROCESSOR_H
#define MITKSTEREODISTORTIONCORRECTIONVIDEOPROCESSOR_H

#include "niftkOpenCVExports.h"
#include "mitkStereoOneTimePointVideoProcessorTemplateMethod.h"

/**
 * class StereoDistortionCorrectionVideoProcessor
 * \brief Derived from StereoVideoProcessorTemplateMethod to correct distortion in interleaved (flicker) stereo stream.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT StereoDistortionCorrectionVideoProcessor : public StereoOneTimePointVideoProcessorTemplateMethod
{

public:

  mitkClassMacro(StereoDistortionCorrectionVideoProcessor, StereoOneTimePointVideoProcessorTemplateMethod);
  mitkNewMacro3Param(StereoDistortionCorrectionVideoProcessor, const bool&, const std::string&, const std::string&);

  /**
   * \brief Call before StereoDistortionCorrectionVideoProcessor::Initialize().
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

  ~StereoDistortionCorrectionVideoProcessor();
  StereoDistortionCorrectionVideoProcessor(const bool& writeInterleaved, CvCapture *capture, CvVideoWriter *writer);
  StereoDistortionCorrectionVideoProcessor(const bool& writeInterleaved, const std::string&, const std::string&);

  StereoDistortionCorrectionVideoProcessor(const StereoDistortionCorrectionVideoProcessor&); // Purposefully not implemented.
  StereoDistortionCorrectionVideoProcessor& operator=(const StereoDistortionCorrectionVideoProcessor&); // Purposefully not implemented.

  virtual void DoProcessing(
      const IplImage &leftInput,
      const IplImage &rightInput,
      IplImage &leftOutput,
      IplImage &rightOutput);

private:

  CvMat *m_IntrinsicLeft;
  CvMat *m_DistortionLeft;
  IplImage *m_MapXLeft;
  IplImage *m_MapYLeft;

  CvMat *m_IntrinsicRight;
  CvMat *m_DistortionRight;
  IplImage *m_MapXRight;
  IplImage *m_MapYRight;

}; // end class

} // end namespace

#endif // MITKSTEREODISTORTIONCORRECTIONVIDEOPROCESSOR_H
