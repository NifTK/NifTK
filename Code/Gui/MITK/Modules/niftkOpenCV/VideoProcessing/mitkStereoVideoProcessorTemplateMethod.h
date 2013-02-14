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
  virtual void Run();

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
   * \brief Derived classes override this method to do their processing.
   */
  virtual void DoProcessing(const IplImage &leftInput, const IplImage &rightInput, IplImage &leftOutput, IplImage &rightOutput) = 0;

private:

  IplImage      *m_LeftInput;
  IplImage      *m_RightInput;
  IplImage      *m_LeftOutput;
  IplImage      *m_RightOutput;
  IplImage      *m_OutputImage;
  bool           m_WriteInterleaved;

}; // end class

} // end namespace

#endif // MITKSTEREOVIDEOPROCESSORTEMPLATEMETHOD_H
