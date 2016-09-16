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
 *
 * Note: These objects are only designed to be initialised with image data once.
 */
class NIFTKIGIDATASOURCES_EXPORT QImageDataType : public IGIDataType
{
public:

  mitkClassMacro(QImageDataType, IGIDataType)
  itkNewMacro(QImageDataType)

  void DeepCopy(const QImage& image);
  void ShallowCopy(const QImage& image);

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  const QImage* GetImage();

protected:

  QImageDataType(); // Purposefully hidden.
  virtual ~QImageDataType(); // Purposefully hidden.

  QImageDataType(const QImageDataType&); // Purposefully not implemented.
  QImageDataType& operator=(const QImageDataType&); // Purposefully not implemented.

private:

  QImage *m_Image;

};

} // end namespace

#endif
