/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkAddOneLayerOfGreyMatterFilter_txx
#define __itkAddOneLayerOfGreyMatterFilter_txx

#include "itkAddOneLayerOfGreyMatterFilter.h"
#include <itkImageRegionConstIteratorWithIndex.h>
#include <niftkConversionUtils.h>

#include <itkLogHelper.h>

namespace itk
{

template <typename TImageType >  
AddOneLayerOfGreyMatterFilter<TImageType>
::AddOneLayerOfGreyMatterFilter()
{
  niftkitkDebugMacro(<<"AddOneLayerOfGreyMatterFilter():Constructed");
}

template <typename TImageType >  
void 
AddOneLayerOfGreyMatterFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TImageType >  
void
AddOneLayerOfGreyMatterFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");
  
  typename InputImageType::ConstPointer inputSegmentedImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));  
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  this->CheckOrAssignLabels();
  
  this->AllocateOutputs();

  niftkitkDebugMacro(<< "GenerateData():Grey label=" << this->GetGreyMatterLabel() \
      << ", white label=" << this->GetWhiteMatterLabel() \
      << ", csf label=" << this->GetExtraCerebralMatterLabel() \
      );
  
  ImageRegionConstIteratorWithIndex<InputImageType> segmentedImageIterator(inputSegmentedImage, inputSegmentedImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

  InputIndexType index;  
  InputPixelType inputValue;
  InputPixelType outputValue;
  
  m_NumberOfGreyInBoundaryLayer = 0;
  m_NumberOfGreyLeftOutsideBoundaryLayer = 0;  
  unsigned long int numberOfWhite = 0;
  
  for (segmentedImageIterator.GoToBegin(),
       outputIterator.GoToBegin();
       !segmentedImageIterator.IsAtEnd();
       ++segmentedImageIterator,
       ++outputIterator)
    {
      outputValue = this->GetExtraCerebralMatterLabel();
      
      inputValue = segmentedImageIterator.Get();
      
      if (inputValue == this->GetGreyMatterLabel())
        {
          index = segmentedImageIterator.GetIndex();
          
          if (this->IsOnWMBoundary(inputSegmentedImage, index, false))
            {
              outputValue = this->GetGreyMatterLabel();
              m_NumberOfGreyInBoundaryLayer++;
            }
          else
            {
              outputValue = this->GetExtraCerebralMatterLabel();
              m_NumberOfGreyLeftOutsideBoundaryLayer++;
            }
        }
      else if (inputValue == this->GetWhiteMatterLabel())
        {
          outputValue = this->GetWhiteMatterLabel();
          numberOfWhite++;
        }
      outputIterator.Set(outputValue);
    }
  niftkitkDebugMacro(<< "GenerateData():Finished, number of white voxels=" << numberOfWhite \
      << ", number of grey in boundary=" <<  m_NumberOfGreyInBoundaryLayer \
      << ", number of grey outside boundary=" << m_NumberOfGreyLeftOutsideBoundaryLayer \
      );
}

} // end namespace

#endif // __itkAddOneLayerOfGreyMatterFilter_txx
