/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKSTEREOTWOTIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H
#define MITKSTEREOTWOTIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H

#include "niftkOpenCVExports.h"
#include "mitkStereoVideoProcessorTemplateMethod.h"

/**
 * class StereoTwoTimePointVideoProcessorTemplateMethod
 * \brief Base class providing stereo processing for two time points.
 * \see StereoVideoProcessorTemplateMethod
 */
namespace mitk {

class NIFTKOPENCV_EXPORT StereoTwoTimePointVideoProcessorTemplateMethod : public StereoVideoProcessorTemplateMethod
{

public:

  mitkClassMacro(StereoTwoTimePointVideoProcessorTemplateMethod, StereoVideoProcessorTemplateMethod);

  /**
   * \see BaseVideoProcessor::Initialize()
   */
  virtual void Initialize();

  /**
   * \brief BaseVideoProcessor::Run()
   */
  virtual void Run();

protected:

  ~StereoTwoTimePointVideoProcessorTemplateMethod();
  StereoTwoTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  StereoTwoTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile);

  StereoTwoTimePointVideoProcessorTemplateMethod(const StereoTwoTimePointVideoProcessorTemplateMethod&); // Purposefully not implemented.
  StereoTwoTimePointVideoProcessorTemplateMethod& operator=(const StereoTwoTimePointVideoProcessorTemplateMethod&); // Purposefully not implemented.

  /**
   * \brief Derived classes override this method to do their processing.
   */
  virtual void DoProcessing(
      const IplImage &leftT1,
      const IplImage &rightT1,
      const IplImage &leftT2,
      const IplImage &rightT2,
      IplImage &leftOutput,
      IplImage &rightOutput) = 0;

private:

  IplImage      *m_LeftT1;
  IplImage      *m_RightT1;
  IplImage      *m_LeftT2;
  IplImage      *m_RightT2;
  IplImage      *m_LeftOutput;
  IplImage      *m_RightOutput;

}; // end class

} // end namespace

#endif // MITKSTEREOTWOTIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H
