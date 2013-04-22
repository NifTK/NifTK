/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkImageConversion_h
#define mitkImageConversion_h

#include "niftkIGIExports.h"
#include <mitkITKImageImport.txx>
#include <opencv2/core/types_c.h>


namespace mitk
{


mitk::Image::Pointer NIFTKIGI_EXPORT CreateMitkImage(const IplImage* image);
mitk::Image::Pointer NIFTKIGI_EXPORT CreateRGBMitkImage(const IplImage* image);
mitk::Image::Pointer NIFTKIGI_EXPORT CreateRGBAMitkImage(const IplImage* image);

} // namespace


#endif // mitkImageConversion_h
