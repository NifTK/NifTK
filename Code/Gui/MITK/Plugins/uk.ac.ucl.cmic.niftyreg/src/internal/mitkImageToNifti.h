/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#ifndef MITKIMAGETONIFTI_HEADER_INCLUDED
#define MITKIMAGETONIFTI_HEADER_INCLUDED

#include "mitkImage.h"
#include "mitkImageDataItem.h"

#include "nifti1_io.h"


/// Create a Nifti image from an mitk::Image

nifti_image *ConvertMitkImageToNifti( mitk::Image::Pointer mitkImage );

template< typename TPixel, unsigned int VImageDimension >
void ConvertMitkImageToNiftiMethod( itk::Image< TPixel, VImageDimension > *itkImage, 
				    nifti_image *niftiImage );

#endif // MITKIMAGETONIFTI_HEADER_INCLUDED

