/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageAndArray_txx
#define __itkImageAndArray_txx

#include "itkImageAndArray.h"

namespace itk
{


/**
 * ImageAndArray()
 */
template<class TPixel, unsigned int VImageDimension>
ImageAndArray<TPixel, VImageDimension>
::ImageAndArray()
{

}


/**
 * Initialize()
 */
template<class TPixel, unsigned int VImageDimension>
void 
ImageAndArray<TPixel, VImageDimension>
::Initialize()
{
  Superclass::Initialize();

}


/**
 * SyncImageAndArray()
 */
template<class TPixel, unsigned int VImageDimension>
void 
ImageAndArray<TPixel, VImageDimension>
::SynchronizeArray()
{
  // Set the array to point to the image (unfortunately
  // itk::Image::Allocate is non-virtual)
  this->SetData( this->GetBufferPointer(), 
		 this->GetLargestPossibleRegion().GetNumberOfPixels(), false );
}


/**
 * PrintSelf(std::ostream& os, Indent indent) const
 */

template<class TPixel, unsigned int VImageDimension>
void 
ImageAndArray<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

}
} // end namespace itk

#endif
