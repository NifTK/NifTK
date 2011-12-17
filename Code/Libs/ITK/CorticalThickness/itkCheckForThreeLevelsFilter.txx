/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkCheckForThreeLevelsFilter_txx
#define __itkCheckForThreeLevelsFilter_txx

#include "itkCheckForThreeLevelsFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "ConversionUtils.h"

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
