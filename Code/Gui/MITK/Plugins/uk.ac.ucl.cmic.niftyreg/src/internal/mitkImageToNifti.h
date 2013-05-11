/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKIMAGETONIFTI_HEADER_INCLUDED
#define MITKIMAGETONIFTI_HEADER_INCLUDED

#include <mitkImage.h>
#include <mitkImageDataItem.h>

#include <nifti1_io.h>


/// Create a Nifti image from an mitk::Image
template<typename NIFTI_PRECISION_TYPE>
nifti_image *ConvertMitkImageToNifti( mitk::Image::Pointer mitkImage );

#ifndef ITK_MANUAL_INSTANTIATION
#include "mitkImageToNifti.txx"
#endif


#endif // MITKIMAGETONIFTI_HEADER_INCLUDED

