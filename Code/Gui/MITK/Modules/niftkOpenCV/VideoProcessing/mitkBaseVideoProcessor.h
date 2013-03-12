/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKBASEVIDEOPROCESSOR_H
#define MITKBASEVIDEOPROCESSOR_H

#include "niftkOpenCVExports.h"

#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

/**
 * \class BaseVideoProcessor
 * \brief Abstract base class that provides methods to read from a capture device,
 * do some processing by calling the virtual Run() method and provides access to a cvVideoWriter object,
 * while managing the memory buffer for each grabbed image.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT BaseVideoProcessor : public itk::Object
{

public:

  mitkClassMacro(BaseVideoProcessor, itk::Object);

  /**
   * \brief This method does any initialization necessary, and derived classes can override
   * it, but must call BaseVideoProcessor::Initialize first within their Initialize method.
   */
  virtual void Initialize();

  /**
   * \brief This is the main method, implemented in derived classes, to run the processing.
   */
  virtual void Run() = 0;

protected:

  ~BaseVideoProcessor();
  BaseVideoProcessor(CvCapture *capture = NULL, CvVideoWriter *writer = NULL);
  BaseVideoProcessor(const std::string& inputFile, const std::string& outputFile);

  BaseVideoProcessor(const BaseVideoProcessor&); // Purposefully not implemented.
  BaseVideoProcessor& operator=(const BaseVideoProcessor&); // Purposefully not implemented.

  /**
   * \brief Derived classes must indicate an output image size, and the derived class
   * must call this->Initialize() before calling this method.
   */
  virtual CvSize GetOutputImageSize() = 0;

  /**
   * \brief Returns the pointer to the current image.
   */
  IplImage* GetCurrentImage() const;

  /**
   * \brief Grabs a new image from the capture device.
   * OpenCV documentation says to NOT try and clear up this memory.
   */
  IplImage* GrabNewImage();

  /**
   * \brief Returns the writer,
   */
  CvVideoWriter* GetWriter() const;

private:

  IplImage      *m_GrabbedImage;
  CvCapture     *m_Capture;
  CvVideoWriter *m_Writer;
  std::string    m_InputFileName;
  std::string    m_OutputFileName;
}; // end class

} // end namespace

#endif // MITKBASEVIDEOPROCESSOR_H
