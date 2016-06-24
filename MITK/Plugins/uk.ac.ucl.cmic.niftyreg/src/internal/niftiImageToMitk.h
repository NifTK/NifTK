/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftiImageToMitk_h
#define niftiImageToMitk_h

#include <mitkImage.h>
#include <mitkImageDataItem.h>

#include <nifti1_io.h>


/// Create an mitk::Image from a Nifti image

mitk::Image::Pointer ConvertNiftiImageToMitk( nifti_image *niftiImage );


#endif // NIFTIIMAGETOMITK_HEADER_INCLUDED

