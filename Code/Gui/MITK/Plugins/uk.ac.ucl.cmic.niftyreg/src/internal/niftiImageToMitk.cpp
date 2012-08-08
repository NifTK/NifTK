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
#include "niftiImageToMitk.h"
#include "mitkBaseProcess.h"
#include "mitkImageAccessByItk.h"


// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitk()
// ---------------------------------------------------------------------------

mitk::Image::Pointer ConvertNiftiImageToMitk( nifti_image *imageNifti )
{

  // Create ITK Image + Geometry

  switch ( imageNifti->dim[0] )
  {
  case 1: 
  {
    return ConvertNiftiImageToMitkDimension< 1 >( imageNifti );
    break;
  }
  case 2: 
  {
    return ConvertNiftiImageToMitkDimension< 2 >( imageNifti );
    break;
  }
  case 3: 
  {
    return ConvertNiftiImageToMitkDimension< 3 >( imageNifti );
    break;
  }
  case 4: 
  {
    return ConvertNiftiImageToMitkDimension< 4 >( imageNifti );
    break;
  }
  case 5: 
  {
    return ConvertNiftiImageToMitkDimension< 5 >( imageNifti );
    break;
  }
  case 6: 
  {
    return ConvertNiftiImageToMitkDimension< 6 >( imageNifti );
    break;
  }
  case 7: 
  {
    return ConvertNiftiImageToMitkDimension< 7 >( imageNifti );
    break;
  }
  default: 
  {
    MITK_ERROR << "Nifti image type not currently " 
	       << "supported for conversion to MITK" << std::endl;
    return 0;
  }
  }

  return 0;
}
      

// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitkDimension()
// ---------------------------------------------------------------------------

template< unsigned int VImageDimension >
mitk::Image::Pointer ConvertNiftiImageToMitkDimension( nifti_image *imageNifti )
{

  // Create ITK Image + Geometry

  switch ( imageNifti->datatype )
  {
  case DT_UINT8: 
  {
    return ConvertNiftiImageToMitkPixel<unsigned char, VImageDimension>( imageNifti );
    break;
  }
  case DT_INT16: 
  {
    return ConvertNiftiImageToMitkPixel<signed short, VImageDimension>( imageNifti );
    break;
  }
  case DT_UINT16: 
  {
    return ConvertNiftiImageToMitkPixel<unsigned short, VImageDimension>( imageNifti );
    break;
  }
  case DT_FLOAT32: 
  {
    return ConvertNiftiImageToMitkPixel<float, VImageDimension>( imageNifti );
    break;
  }
  case DT_FLOAT64: 
  {
    return ConvertNiftiImageToMitkPixel<double, VImageDimension>( imageNifti );
    break;
  }
  default: 
  {
    MITK_ERROR << "Nifti image type not currently " 
	       << "supported for conversion to MITK" << std::endl;
    return 0;
  }
  }

  return 0;
}
      

// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitkPixel()
// ---------------------------------------------------------------------------

template< typename TPixel, unsigned int VImageDimension >
mitk::Image::Pointer ConvertNiftiImageToMitkPixel( nifti_image *imageNifti )
{
  unsigned int i;

  typedef itk::Image<TPixel, VImageDimension> ImageType;

  typename ImageType::RegionType myRegion;
  typename ImageType::SizeType mySize;
  typename ImageType::IndexType myIndex;
  typename ImageType::SpacingType mySpacing;

  typename ImageType::Pointer imageITK = ImageType::New();

  // Create new, empty MITK image
  typename mitk::Image::Pointer imageMITK = mitk::Image::New();


  for ( i=1; i<= VImageDimension; i++ ) 
  {
    myIndex[ i - 1 ] = 0;
    mySize[ i - 1 ] = imageNifti->dim[i]; // no. of voxels in dimension i

    mySpacing[ i - 1 ] = imageNifti->pixdim[i];
  }

  myRegion.SetIndex( myIndex );
  myRegion.SetSize( mySize);

  imageITK->SetSpacing(mySpacing);
  imageITK->SetRegions( myRegion);

  imageITK->Allocate();
  imageITK->FillBuffer(0);


  TPixel *pNiftiPixels = static_cast<TPixel*>( imageNifti->data );

  itk::ImageRegionIterator<ImageType>  iterator(imageITK, imageITK->GetLargestPossibleRegion());

  iterator.GoToBegin();

  while (!iterator.IsAtEnd())
  {
    iterator.Set( *pNiftiPixels );
    ++iterator;
    pNiftiPixels++;
  }

  imageITK->Print( std::cout );

  mitk::CastToMitkImage( imageITK, imageMITK );

  return imageMITK;
}


