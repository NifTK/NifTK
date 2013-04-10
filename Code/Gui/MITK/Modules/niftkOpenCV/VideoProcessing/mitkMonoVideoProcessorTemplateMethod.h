/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMONOVIDEOPROCESSORTEMPLATEMETHOD_H
#define MITKMONOVIDEOPROCESSORTEMPLATEMETHOD_H

#include "niftkOpenCVExports.h"
#include "mitkBaseVideoProcessor.h"

/**
 * \class MonoVideoProcessorTemplateMethod
 * \brief Base class providing methods to read images from a capture device,
 * process each one sequentially, and then write to the video writer.
 * \see BaseVideoProcessor
 */
namespace mitk {

class NIFTKOPENCV_EXPORT MonoVideoProcessorTemplateMethod : public BaseVideoProcessor
{

public:

  mitkClassMacro(MonoVideoProcessorTemplateMethod, BaseVideoProcessor);

  /**
   * \see BaseVideoProcessor::Initialize()
   */
  virtual void Initialize();

  /**
   * \see BaseVideoProcessor::Run()
   */
  virtual void Run();

protected:

  ~MonoVideoProcessorTemplateMethod();
  MonoVideoProcessorTemplateMethod(CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  MonoVideoProcessorTemplateMethod(const std::string& inputFile, const std::string& outputFile);

  MonoVideoProcessorTemplateMethod(const MonoVideoProcessorTemplateMethod&); // Purposefully not implemented.
  MonoVideoProcessorTemplateMethod& operator=(const MonoVideoProcessorTemplateMethod&); // Purposefully not implemented.

  /**
   * \brief Overrides base class method \see BaseVideoProcessor::GetOutputImageSize(), returning the size
   * of the grabbed image in the base class. i.e. this class does nothing to the size.
   */
  CvSize GetOutputImageSize();

  /**
   * \brief Derived classes override this method to do their processing.
   */
  virtual void DoProcessing(const IplImage &input, IplImage &output) = 0;

private:

  IplImage *m_OutputImage;

}; // end class

} // end namespace

#endif // MITKMONOVIDEOPROCESSORTEMPLATEMETHOD_H
