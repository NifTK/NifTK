/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQImageConversion.h"
#include <niftkImageConversion.h>

namespace niftk
{

//-----------------------------------------------------------------------------
mitk::Image::Pointer CreateMitkImage(const QImage* image,
                                     unsigned int& outputNumberOfBytes)
{
  QImage *imageToConvert = const_cast<QImage*>(image);

  QImage tmp;
  if (   image->format() != QImage::Format_Indexed8
      && image->format() != QImage::Format_RGB888
#if QT_VERSION >= QT_VERSION_CHECK(5, 2, 0)
      && image->format() != QImage::Format_RGBA8888
#endif
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
      && image->format() != QImage::Format_Grayscale8
#endif
      )
  {
    tmp = image->convertToFormat(QImage::Format_RGB888);
    imageToConvert = &tmp;
  }

  outputNumberOfBytes = imageToConvert->byteCount();

  // FIXME: check for channel layout: rgb vs bgr
  switch (imageToConvert->format())
  {
    case QImage::Format_Indexed8:
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
    case QImage::Format_Grayscale8:
#endif
      return CreateMitkImageInternal<unsigned char>(reinterpret_cast<const char*>(imageToConvert->bits()),
                                                    1,
                                                    imageToConvert->width(),
                                                    imageToConvert->width(),
                                                    imageToConvert->height()
                                                    );
    case QImage::Format_RGB888:
      return CreateMitkImageInternal<UCRGBPixelType>(reinterpret_cast<const char*>(imageToConvert->bits()),
                                                     3,
                                                     imageToConvert->width(),
                                                     imageToConvert->width() * 3,
                                                     imageToConvert->height()
                                                     );
#if QT_VERSION >= QT_VERSION_CHECK(5, 2, 0)
    case QImage::Format_RGBA8888:
      return CreateMitkImageInternal<UCRGBAPixelType>(reinterpret_cast<const char*>(imageToConvert->bits()),
                                                      4,
                                                      imageToConvert->width(),
                                                      imageToConvert->width() * 4,
                                                      imageToConvert->height()
                                                      );
#endif
  }

  assert(false);
  return 0;
}

} // namespace
