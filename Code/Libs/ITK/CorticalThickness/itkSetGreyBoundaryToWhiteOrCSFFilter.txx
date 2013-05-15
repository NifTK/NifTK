/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSetGreyBoundaryToWhiteOrCSFFilter_txx
#define __itkSetGreyBoundaryToWhiteOrCSFFilter_txx

#include "itkSetGreyBoundaryToWhiteOrCSFFilter.h"
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <ConversionUtils.h>

#include <itkLogHelper.h>

namespace itk
{
template <typename TImageType, typename TScalarType, unsigned int NDimensions>  
SetGreyBoundaryToWhiteOrCSFFilter<TImageType, TScalarType, NDimensions>
::SetGreyBoundaryToWhiteOrCSFFilter()
{
  m_ExpectedVoxelSize = 1;
  m_TaggedCSFLabel = 4;
  niftkitkDebugMacro(<<"SetGreyBoundaryToWhiteOrCSFFilter():Constructed");
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions> 
void 
SetGreyBoundaryToWhiteOrCSFFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "ExpectedVoxelSize = " << m_ExpectedVoxelSize << std::endl;
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions>   
void
SetGreyBoundaryToWhiteOrCSFFilter<TImageType, TScalarType, NDimensions>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");

  this->AllocateOutputs();

  typename InputImageType::ConstPointer inputLabelImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));  
  typename InputImageType::ConstPointer oneLayerImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));
  typename ThicknessImageType::ConstPointer thicknessImage = static_cast< ThicknessImageType * >(this->ProcessObject::GetInput(2));
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  typename itk::ImageRegionConstIterator<InputImageType> inputLabelImageIterator(inputLabelImage, inputLabelImage->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<InputImageType> oneLayerImageIterator(oneLayerImage, oneLayerImage->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<ThicknessImageType> thicknessImageIterator(thicknessImage, thicknessImage->GetLargestPossibleRegion());
  typename itk::ImageRegionIterator<InputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

  unsigned long int numberOfGreyToCSF = 0;
  unsigned long int numberOfGreyToWhite = 0;
  unsigned long int numberOfWhite = 0;
  unsigned long int numberOfCSF = 0;
  m_NumberOfGreyBefore = 0;
  m_NumberOfGreyAfter = 0;
  
  InputPixelType labelInput;
  InputPixelType oneLayerInput;
  InputPixelType output;
  InputIndexType index;
  
  for (inputLabelImageIterator.GoToBegin(),
      oneLayerImageIterator.GoToBegin(),
      thicknessImageIterator.GoToBegin(),
      outputIterator.GoToBegin();
       !inputLabelImageIterator.IsAtEnd();
       ++outputIterator,
       ++thicknessImageIterator,
       ++oneLayerImageIterator,
       ++inputLabelImageIterator
       )
    {
      index = inputLabelImageIterator.GetIndex();
      labelInput = inputLabelImageIterator.Get();
      
      output = labelInput;
      
      if (labelInput == this->GetWhiteMatterLabel())
        {
          output = labelInput;
          numberOfWhite++;
        }
      else if (labelInput == this->GetExtraCerebralMatterLabel())
        {
          output = labelInput;
          numberOfCSF++;
        }
      else
        {
          // Else, so it must be a grey voxel.
          m_NumberOfGreyBefore++;
          
          // If we are looking at the one voxel wide boundary.
          oneLayerInput = oneLayerImageIterator.Get();
          
          if (oneLayerInput == this->GetGreyMatterLabel())
            {
              if (thicknessImageIterator.Get() <= m_ExpectedVoxelSize)
                {
                  output = this->GetWhiteMatterLabel();
                  numberOfGreyToWhite++;
                }
              else 
                {
                  output = m_TaggedCSFLabel;
                  numberOfGreyToCSF++;
                }
            }
          else
            {
              // otherwise, its GM, but not on the boundary.
              output = labelInput;
              m_NumberOfGreyAfter++;
            }
        }
      
      outputIterator.Set(output);
      
    } // end for each voxel
  
  niftkitkDebugMacro(<<"GenerateData():Thickness threshold=" << m_ExpectedVoxelSize \
      << ", number of white=" << numberOfWhite \
      << ", number of csf=" << numberOfCSF \
      << ", number of grey before=" << m_NumberOfGreyBefore \
      << ", number of grey after=" << m_NumberOfGreyAfter \
      << ", number of grey switched to csf=" << numberOfGreyToCSF \
      << ", number of grey switched to white=" << numberOfGreyToWhite \
      );
   
}

} // end namespace

#endif // __itkSetGreyBoundaryToWhiteOrCSFFilter_txx
