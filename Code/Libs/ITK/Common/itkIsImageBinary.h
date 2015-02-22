/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkIsImageBinary_h
#define __itkIsImageBinary_h

#include <itkImageRegionConstIterator.h>


namespace itk
{

//  --------------------------------------------------------------------------
/// Return whether an image is binary or not
//  --------------------------------------------------------------------------

template < typename TImage >
bool
IsImageBinary( typename TImage::Pointer image )
{
  typedef typename TImage::PixelType InputPixelType;
  
  InputPixelType intensity1;
  InputPixelType intensity2;

  itk::ImageRegionIterator< TImage > 
    itImage( image, image->GetLargestPossibleRegion() );

  itImage.GoToBegin();

  // Get the first pixel's intensity

  if ( ! itImage.IsAtEnd() )
  {
    intensity1 = itImage.Get();
    ++itImage;
  }
  else 
  {
    return false;
  }


  // Get the next pixel with a different intensity

  while ( ( ! itImage.IsAtEnd() ) && ( itImage.Get() == intensity1 ) )
  {
    ++itImage;
  }

  if ( ! itImage.IsAtEnd() )
  {
    intensity2 = itImage.Get();
    ++itImage;
  }
  else 
  {
    return false;
  }


  // Are any pixels not equal to these two intensities?

  for ( ; 
        ! itImage.IsAtEnd(); 
        ++itImage )
  {
    if ( ( itImage.Get() != intensity1 ) && 
         ( itImage.Get() != intensity2 ) )
    {
      return false;
    }
  }

  return true;
}


} // end namespace itk

#endif /* __itkIsImageBinary_h */
