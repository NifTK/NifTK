/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSetBoundaryVoxelsToValueFilter_txx
#define __itkSetBoundaryVoxelsToValueFilter_txx

#include <itkShapedNeighborhoodIterator.h>
#include <itkNeighborhoodAlgorithm.h>
#include "itkSetBoundaryVoxelsToValueFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkProgressReporter.h>
#include <niftkConversionUtils.h>

#include <itkLogHelper.h>


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
SetBoundaryVoxelsToValueFilter<TInputImage,TOutputImage>
::SetBoundaryVoxelsToValueFilter()
{
  m_Value = 0.;
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
SetBoundaryVoxelsToValueFilter<TInputImage,TOutputImage>
::GenerateData( )
{
  this->BeforeThreadedGenerateData();

  // Call a method that can be overriden by a subclass to allocate
  // memory for the filter's outputs
  this->AllocateOutputs();

  OutputImageIndexType outIndex;
  OutputImageSizeType outSize;

  // Obtain image pointers

  InputImageConstPointer inImage  = this->GetInput();
  OutputImagePointer     outImage = this->GetOutput();

  outSize = outImage->GetLargestPossibleRegion().GetSize();

  ImageRegionConstIterator< InputImageType >  inIterator(inImage,  inImage->GetLargestPossibleRegion());
  ImageRegionIterator< OutputImageType >     outIterator(outImage, outImage->GetLargestPossibleRegion());

  while ( ! outIterator.IsAtEnd() ) {

    outIndex = outIterator.GetIndex();

    if (   (outIndex[0] == 0) 
        || (outIndex[1] == 0) 
        || (outIndex[2] == 0)
        || (outIndex[0] == (int)(outSize[0]) - 1) 
        || (outIndex[1] == (int)(outSize[1]) - 1) 
        || (outIndex[2] == (int)(outSize[2]) - 1))
    
      outIterator.Set( m_Value );

    else
      outIterator.Set( inIterator.Get() );

    ++outIterator;
    ++inIterator;
  }

  // Call a method that can be overridden by a subclass to perform
  // some calculations after all the threads have completed
  this->AfterThreadedGenerateData();
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
SetBoundaryVoxelsToValueFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << "Insertion value: " << m_Value << std::endl;
}

} // end namespace itk

#endif
