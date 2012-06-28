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



// ITK
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"

// MITK
#include "mitkImageToNifti.h"
#include "mitkBaseProcess.h"
#include "mitkImageAccessByItk.h"


// ---------------------------------------------------------------------------
// ConvertMitkImageToNifti()
// ---------------------------------------------------------------------------

nifti_image *ConvertMitkImageToNifti( mitk::Image::Pointer mitkImage )
{
  int i, nBytesPerVoxel, swapsize;

  // Check the input MITK image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ( mitkImage->GetDimension() < 2 ) ||
       ( mitkImage->GetDimension() > 7 ) ) 
  {
    MITK_ERROR << "Cannot convert MITK image of dimension: " 
	       << mitkImage->GetDimension() << " to nifti" << std::endl;
    return 0;
  }

  if ( mitkImage->GetPixelType().GetNumberOfComponents() != 1 ) 
  {
    MITK_ERROR << "Conversion of " 
	       << mitkImage->GetPixelType().GetNumberOfComponents() 
	       << " component MITK image not currently supported" << std::endl;
    return 0;
  }
      

  // Create the nifti header
  // ~~~~~~~~~~~~~~~~~~~~~~~

  struct nifti_1_header niftiHeader;

  // zero out header, to be safe
  memset( &niftiHeader, 0, sizeof(niftiHeader) );
  
  niftiHeader.sizeof_hdr = sizeof(niftiHeader);

  niftiHeader.regular    = 'r';

  // The image dimension
  niftiHeader.dim[0] = mitkImage->GetDimension(); 

  // Set the number of voxels
  for ( i=1; i<= niftiHeader.dim[0]; i++ )
    niftiHeader.dim[i] = mitkImage->GetDimension( i - 1 ); // no. of voxels in dimension i

  // Set the voxel resolution
  niftiHeader.pixdim[0] = 0.0;	// Undefined

  for ( i=1; i<= niftiHeader.dim[0]; i++ ) 
    niftiHeader.pixdim[i] = mitkImage->GetGeometry()->GetSpacing()[i-1];

  // Set the data type
  if ( (mitkImage->GetPixelType().GetTypeId() == typeid(unsigned char))
       && (mitkImage->GetPixelType().GetBitsPerComponent() == 8) )
  {
    niftiHeader.datatype = DT_UINT8;
  } 
  else if ( (mitkImage->GetPixelType().GetTypeId() == typeid(signed short))
	   && (mitkImage->GetPixelType().GetBitsPerComponent() == 16) )
  {
    niftiHeader.datatype = DT_INT16;
  } 
  else if ( (mitkImage->GetPixelType().GetTypeId() == typeid(unsigned short)) 
	   && (mitkImage->GetPixelType().GetBitsPerComponent() == 16) )
  {
    niftiHeader.datatype = DT_UINT16;
  }
  else if ( (mitkImage->GetPixelType().GetTypeId() == typeid(float)) 
	   && (mitkImage->GetPixelType().GetBitsPerComponent() == 32) )
  {
    niftiHeader.datatype = DT_FLOAT32;
  } 
  else if ( (mitkImage->GetPixelType().GetTypeId()== typeid(double)) 
	   && (mitkImage->GetPixelType().GetBitsPerComponent() == 64) ) 
  {
    niftiHeader.datatype = DT_FLOAT64;
  }
  else 
  {
    MITK_ERROR << "MITK image type not currently " 
	       << "supported for conversion to nifti" << std::endl;
    return 0;
  }

  nifti_datatype_sizes( niftiHeader.datatype, &nBytesPerVoxel, &swapsize );

  niftiHeader.bitpix = 8*nBytesPerVoxel; // Number bits/voxel

  // init to single file
  strcpy( niftiHeader.magic, "n+1" ); 	// NIFTI-1 flag

  if ( niftiHeader.dim[1] < 1 ) niftiHeader.dim[1] = 1;
  if ( niftiHeader.dim[2] < 1 ) niftiHeader.dim[2] = 1;
  if ( niftiHeader.dim[3] < 1 ) niftiHeader.dim[3] = 1;
  if ( niftiHeader.dim[4] < 1 ) niftiHeader.dim[4] = 1;
  if ( niftiHeader.dim[5] < 1 ) niftiHeader.dim[5] = 1;
  if ( niftiHeader.dim[6] < 1 ) niftiHeader.dim[6] = 1;
  if ( niftiHeader.dim[7] < 1 ) niftiHeader.dim[7] = 1;
				                        
  if ( niftiHeader.scl_slope == 0 ) niftiHeader.scl_slope = 1.f;


  // Allocate the nifti image
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *niftiImage = 0;
  
  niftiImage = nifti_convert_nhdr2nim( niftiHeader, NULL );
  niftiImage->data = calloc( niftiImage->nvox, niftiImage->nbyper );

  niftiImage->fname = NULL;
  niftiImage->iname = NULL;


  // Copy the voxel intensity data across
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  try
  {
    AccessByItk_n( mitkImage, ConvertMitkImageToNiftiMethod, (niftiImage) );
  }

  catch (const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "During ImageStatisticsView::Update, "
	       << "caught mitk::AccessByItkException caused by:" << e.what() << std::endl;
  }

  catch( itk::ExceptionObject &err )
  {
    MITK_ERROR << "During ImageStatisticsView::Update, "
	       << "caught itk::ExceptionObject caused by:" << err.what() << std::endl;
  }


  return niftiImage;
}


// ---------------------------------------------------------------------------
// ConvertMitkImageToNiftiMethod()
// ---------------------------------------------------------------------------

template<typename TPixel, unsigned int VImageDimension>
void ConvertMitkImageToNiftiMethod(itk::Image<TPixel, VImageDimension>* itkImage, 
				   nifti_image *niftiImage)
{
  TPixel *pNiftiPixels = static_cast<TPixel*>( niftiImage->data );

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  
  itk::ImageRegionIterator<ImageType> inputIterator( itkImage, 
						     itkImage->GetLargestPossibleRegion() );

  for ( inputIterator.GoToBegin(); ! inputIterator.IsAtEnd(); ++inputIterator )
  {
    *pNiftiPixels = inputIterator.Get();
    pNiftiPixels++;
  }
}
