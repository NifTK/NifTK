/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkImageConversion_h
#define niftkImageConversion_h

#include "niftkIGIExports.h"
#include <mitkITKImageImport.txx>
#include <opencv2/core/types_c.h>


namespace niftk
{


mitk::Image::Pointer NIFTKIGI_EXPORT CreateMitkImage(const IplImage* image);


} // namespace


#endif // niftkImageConversion_h
