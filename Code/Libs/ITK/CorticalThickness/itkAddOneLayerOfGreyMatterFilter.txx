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
#ifndef __itkAddOneLayerOfGreyMatterFilter_txx
#define __itkAddOneLayerOfGreyMatterFilter_txx

#include "itkAddOneLayerOfGreyMatterFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "ConversionUtils.h"

#include "itkLogHelper.h"

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
