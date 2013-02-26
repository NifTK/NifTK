/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKSTEREOVIDEOPROCESSORTEMPLATEMETHOD_H
#define MITKSTEREOVIDEOPROCESSORTEMPLATEMETHOD_H

#include "niftkOpenCVExports.h"
#include "mitkBaseVideoProcessor.h"

/**
 * class StereoVideoProcessorTemplateMethod
 * \brief Base class providing methods to read images from a capture device,
 * that provides flicker stereo (interleaved left, then right, then left, then right),
 * process pairs sequentially, and then write to the video writer.
 * \see BaseVideoProcessor
 */
namespace mitk {

class NIFTKOPENCV_EXPORT StereoVideoProcessorTemplateMethod : public BaseVideoProcessor
{

public:

  mitkClassMacro(StereoVideoProcessorTemplateMethod, BaseVideoProcessor);

  /**
   * \see BaseVideoProcessor::Initialize()
   */
  virtual void Initialize();

  /**
   * \brief BaseVideoProcessor::Run()
   */
  virtual void Run() = 0;

protected:

  ~StereoVideoProcessorTemplateMethod();
  StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  StereoVideoProcessorTemplateMethod(const bool& writeInterleaved, const std::string& inputFile, const std::string& outputFile);

  StereoVideoProcessorTemplateMethod(const StereoVideoProcessorTemplateMethod&); // Purposefully not implemented.
  StereoVideoProcessorTemplateMethod& operator=(const StereoVideoProcessorTemplateMethod&); // Purposefully not implemented.

  /**
   * \brief Overrides base class method \see BaseVideoProcessor::GetOutputImageSize(), returning either
   * the size of the grabbed image, or if writerInterleaved constructor argument is true, the size of the
   * left and right image side by side.
   */
  CvSize GetOutputImageSize();

  /**
   * \brief Utility method, to facilitate writing out a stereo pair.
   */
  virtual void WriteOutput(IplImage &leftOutput, IplImage &rightOutput);

private:

  bool           m_WriteInterleaved;
  IplImage      *m_OutputImage;
  int            m_FrameCount;

}; // end class

} // end namespace

#endif // MITKSTEREOVIDEOPROCESSORTEMPLATEMETHOD_H
