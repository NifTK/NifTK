/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQImageConversion_h
#define niftkQImageConversion_h

#include "niftkQImageConversionExports.h"
#include <mitkImage.h>
#include <QImage>

namespace niftk
{

/**
* Supports RGB, RGBA and grayscale images, currently 8-bit per channel only!.
* Known bug: does not take care of different channel layouts: BGR vs RGB!
*/
mitk::Image::Pointer NIFTKQIMAGECONVERSION_EXPORT CreateMitkImage(const QImage* image);

} // namespace

#endif
