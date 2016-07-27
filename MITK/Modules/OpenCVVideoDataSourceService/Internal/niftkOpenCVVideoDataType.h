/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOpenCVVideoDataType_h
#define niftkOpenCVVideoDataType_h

#include <niftkIGIDataType.h>
#include <cv.h>

namespace niftk
{

/**
 * \class OpenCVVideoDataType
 * \brief Class to represent video frame data from OpenCV.
 */
class OpenCVVideoDataType : public IGIDataType
{
public:

  mitkClassMacro(OpenCVVideoDataType, IGIDataType)
  itkNewMacro(OpenCVVideoDataType)

  /**
  * \brief Used for loading in an image.
  */
  void CloneImage(const IplImage *image);

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  const IplImage* GetImage();

protected:

  OpenCVVideoDataType(); // Purposefully hidden.
  virtual ~OpenCVVideoDataType(); // Purposefully hidden.

  OpenCVVideoDataType(const OpenCVVideoDataType&); // Purposefully not implemented.
  OpenCVVideoDataType& operator=(const OpenCVVideoDataType&); // Purposefully not implemented.

private:

  IplImage *m_Image;

};

} // end namespace

#endif
