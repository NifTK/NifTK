/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCheckForThreeLevelsFilter_txx
#define __itkCheckForThreeLevelsFilter_txx

#include "itkCheckForThreeLevelsFilter.h"
#include <itkImageRegionConstIteratorWithIndex.h>
#include <ConversionUtils.h>

namespace itk
{

template <typename TImageType >  
CheckForThreeLevelsFilter<TImageType>
::CheckForThreeLevelsFilter()
{
  niftkitkDebugMacro(<<"CheckForThreeLevelsFilter():Constructed");
}

template <typename TImageType >  
void 
CheckForThreeLevelsFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TImageType >  
void
CheckForThreeLevelsFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");
  
  typename InputImageType::ConstPointer inputSegmentedImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));  
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  this->CheckInputsAndOutputsSameSize();

  this->CheckOrAssignLabels();
  
  this->AllocateOutputs();

  m_ListOfGreyMatterVoxels.clear();
  
  ImageRegionConstIteratorWithIndex<InputImageType> segmentedIterator(inputSegmentedImage, inputSegmentedImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType>     outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  
  segmentedIterator.GoToBegin();
  outputIterator.GoToBegin();
  
  while(!segmentedIterator.IsAtEnd() && !outputIterator.IsAtEnd())
    {
      outputIterator.Set(segmentedIterator.Get());
      ++segmentedIterator;
      ++outputIterator;
    }
  
  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
