/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-06-28 11:43:10 +0100 (Thu, 28 Jun 2012) $
 Revision          : $Revision: 9264 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#ifndef NIFTIIMAGETOMITK_HEADER_INCLUDED
#define NIFTIIMAGETOMITK_HEADER_INCLUDED

#include "mitkImage.h"
#include "mitkImageDataItem.h"

#include "nifti1_io.h"


/// Create an mitk::Image from a Nifti image

mitk::Image::Pointer ConvertNiftiImageToMitk( nifti_image *niftiImage );


#endif // NIFTIIMAGETOMITK_HEADER_INCLUDED

