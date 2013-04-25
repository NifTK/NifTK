/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkIGIOpenCVDataType_h
#define mitkIGIOpenCVDataType_h

#include "niftkIGIExports.h"
#include "mitkIGIDataType.h"
#include <cv.h>

namespace mitk
{

/**
 * \class IGIOpenCVDataType
 * \brief Class to represent video frame data from OpenCV, to integrate within the niftkIGI framework.
 */
class NIFTKIGI_EXPORT IGIOpenCVDataType : public IGIDataType
{
public:

  mitkClassMacro(IGIOpenCVDataType, IGIDataType);
  itkNewMacro(IGIOpenCVDataType);

  /**
   * \brief Used for loading in an image, see mitk::OpenCVVideoSource
   */
  void CloneImage(const IplImage *image);

  /**
   * \brief Returns the internal image, so do not modify it.
   */
  const IplImage* GetImage();

protected:

  IGIOpenCVDataType(); // Purposefully hidden.
  virtual ~IGIOpenCVDataType(); // Purposefully hidden.

  IGIOpenCVDataType(const IGIOpenCVDataType&); // Purposefully not implemented.
  IGIOpenCVDataType& operator=(const IGIOpenCVDataType&); // Purposefully not implemented.

private:

  IplImage *m_Image;

};

} // end namespace

#endif
