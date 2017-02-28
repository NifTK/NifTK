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

  /**
   * \brief Default constructor, so internal image is null.
   */
  OpenCVVideoDataType();

  /**
  * \brief If you provide an image, this OpenCVVideoDataType does a shallow copy of just pointer.
  */
  OpenCVVideoDataType(IplImage *image);

  virtual ~OpenCVVideoDataType();
  OpenCVVideoDataType(const OpenCVVideoDataType&);             // Copy constructor
  OpenCVVideoDataType& operator=(const OpenCVVideoDataType&);  // Copy assignment
  OpenCVVideoDataType(OpenCVVideoDataType&&);                  // Move constructor
  OpenCVVideoDataType& operator=(OpenCVVideoDataType&&);       // Move assignment

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  const IplImage* GetImage() const;

  /**
  * \brief Clones/Copies the provided image;
  */
  void SetImage(const IplImage *image);

  /**
   * \brief Overrides base class, but only copies OpenCVVideoDataType.
   */
  virtual void Clone(const IGIDataType&) override;

private:

  void CloneImage(const IplImage *image);
  IplImage *m_Image;

};

} // end namespace

#endif
