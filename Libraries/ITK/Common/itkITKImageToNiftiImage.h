/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkITKImageToNiftiImage_h
#define itkITKImageToNiftiImage_h

#include <itkImage.h>
#include <nifti1_io.h>


/// Create a Nifti image from an itk::Image

ITK_EXPORT  
template< typename NIFTI_PRECISION_TYPE, typename ITK_VOXEL_TYPE, unsigned int VImageDimension>
nifti_image *ConvertITKImageToNiftiImage( typename itk::Image< ITK_VOXEL_TYPE, VImageDimension >::Pointer itkImage );

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkITKImageToNiftiImage.txx"
#endif


#endif // itkITKImageToNiftiImage_h

