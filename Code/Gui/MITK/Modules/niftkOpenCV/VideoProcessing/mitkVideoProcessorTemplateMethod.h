/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKVIDEOPROCESSORTEMPLATEMETHOD_H
#define MITKVIDEOPROCESSORTEMPLATEMETHOD_H

#include "niftkOpenCVExports.h"

#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

/**
 * class VideoProcessorTemplateMethod
 * \brief Uses Template method to read from a capture device, do some processing,
 * and write to a cvVideoWriter object.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT VideoProcessorTemplateMethod : public itk::Object
{

public:

  mitkClassMacro(VideoProcessorTemplateMethod, itk::Object);

  void Run();

protected:

  ~VideoProcessorTemplateMethod();
  VideoProcessorTemplateMethod(CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  VideoProcessorTemplateMethod(const std::string& inputFile, const std::string& outputFile = std::string());

  VideoProcessorTemplateMethod(const VideoProcessorTemplateMethod&); // Purposefully not implemented.
  VideoProcessorTemplateMethod& operator=(const VideoProcessorTemplateMethod&); // Purposefully not implemented.

  virtual void DoProcessing(const IplImage &input, IplImage &output) = 0;

  IplImage* GetImage();

private:

  CvCapture     *m_Capture;
  CvVideoWriter *m_Writer;

}; // end class

} // end namespace

#endif // MITKVIDEOPROCESSINGFACADE_H
