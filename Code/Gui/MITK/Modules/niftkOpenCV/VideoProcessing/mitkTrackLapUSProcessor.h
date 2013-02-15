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
#include "mitkStereoVideoProcessorTemplateMethod.h"

/**
 * class TrackLapUSProcessor
 * \brief Derived from StereoVideoProcessorTemplateMethod to track Laparoscopic Ultrasound in interleaved (flicker) stereo stream.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT TrackLapUSProcessor : public StereoVideoProcessorTemplateMethod
{

public:

  mitkClassMacro(TrackLapUSProcessor, StereoVideoProcessorTemplateMethod);
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

  virtual void DoProcessing(const IplImage &leftInput, const IplImage &rightInput, IplImage &leftOutput, IplImage &rightOutput);

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
