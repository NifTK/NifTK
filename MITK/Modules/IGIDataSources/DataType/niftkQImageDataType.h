/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQImageDataType_h
#define niftkQImageDataType_h

#include <niftkIGIDataType.h>
#include <niftkIGIDataSourcesExports.h>
#include <QImage>

namespace niftk
{

/**
 * \class QImageDataType
 * \brief Class to represent a frame of video/ultrasound data using QImage.
 */
class NIFTKIGIDATASOURCES_EXPORT QImageDataType : public IGIDataType
{
public:

  /**
   * \brief Default constructor, so internal image is null.
   */
  QImageDataType();

  /**
  * \brief If you provide an image, this QImageDataType does a shallow copy of just pointer.
  */
  QImageDataType(QImage *image);

  virtual ~QImageDataType();
  QImageDataType(const QImageDataType&);             // Copy constructor
  QImageDataType& operator=(const QImageDataType&);  // Copy assignment
  QImageDataType(QImageDataType&&);                  // Move constructor
  QImageDataType& operator=(QImageDataType&&);       // Move assignment

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  const QImage* GetImage() const;

  /**
  * \brief Clones/Copies the provided image;
  */
  void SetImage(const QImage *image);

  /**
  * \brief Overrides base class, but only copies QImageDataType.
  */
  virtual void Clone(const IGIDataType&) override;

private:

  void CloneImage(const QImage *image);
  QImage *m_Image;

};

} // end namespace

#endif
