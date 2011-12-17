/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-24 16:54:16 +0100 (Fri, 24 Sep 2010) $
 Revision          : $Revision: 3944 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
