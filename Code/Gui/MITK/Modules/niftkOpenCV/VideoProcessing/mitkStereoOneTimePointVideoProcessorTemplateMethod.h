/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKSTEREOONETIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H
#define MITKSTEREOONETIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H

#include "niftkOpenCVExports.h"
#include "mitkStereoVideoProcessorTemplateMethod.h"

/**
 * class StereoOneTimePointVideoProcessorTemplateMethod
 * \brief Base class providing stereo processing for a single time point.
 * \see StereoVideoProcessorTemplateMethod
 */
namespace mitk {

class NIFTKOPENCV_EXPORT StereoOneTimePointVideoProcessorTemplateMethod : public StereoVideoProcessorTemplateMethod
{

public:

  mitkClassMacro(StereoOneTimePointVideoProcessorTemplateMethod, StereoVideoProcessorTemplateMethod);

  /**
   * \see BaseVideoProcessor::Initialize()
   */
  virtual void Initialize();

  /**
   * \brief BaseVideoProcessor::Run()
   */
  virtual void Run();

protected:

  ~StereoOneTimePointVideoProcessorTemplateMethod();
  StereoOneTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  StereoOneTimePointVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile);

  StereoOneTimePointVideoProcessorTemplateMethod(const StereoOneTimePointVideoProcessorTemplateMethod&); // Purposefully not implemented.
  StereoOneTimePointVideoProcessorTemplateMethod& operator=(const StereoOneTimePointVideoProcessorTemplateMethod&); // Purposefully not implemented.

  /**
   * \brief Derived classes override this method to do their processing.
   */
  virtual void DoProcessing(const IplImage &leftInput, const IplImage &rightInput, IplImage &leftOutput, IplImage &rightOutput) = 0;

private:

  IplImage      *m_LeftInput;
  IplImage      *m_RightInput;
  IplImage      *m_LeftOutput;
  IplImage      *m_RightOutput;

}; // end class

} // end namespace

#endif // MITKSTEREOONETIMEPOINTVIDEOPROCESSORTEMPLATEMETHOD_H
