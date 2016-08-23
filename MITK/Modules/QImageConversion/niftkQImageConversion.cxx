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
mitk::Image::Pointer CreateMitkImage(const QImage* image)
{
  // FIXME: check for channel layout: rgb vs bgr
  switch (image->format())
  {
    case QImage::Format_Indexed8:
      return CreateMitkImageInternal<unsigned char>(reinterpret_cast<const char*>(image->bits()),
                                                    1,
                                                    image->width(),
                                                    image->width(),
                                                    image->height()
                                                    );
    case QImage::Format_RGB888:
      return CreateMitkImageInternal<UCRGBPixelType>(reinterpret_cast<const char*>(image->bits()),
                                                     3,
                                                     image->width(),
                                                     image->width() * 3,
                                                     image->height()
                                                     );
    case QImage::Format_RGBA8888:
      return CreateMitkImageInternal<UCRGBAPixelType>(reinterpret_cast<const char*>(image->bits()),
                                                      4,
                                                      image->width(),
                                                      image->width() * 4,
                                                      image->height()
                                                      );
  }

  assert(false);
  return 0;
}

} // namespace
