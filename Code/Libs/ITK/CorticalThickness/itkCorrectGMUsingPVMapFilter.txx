/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCorrectGMUsingPVMapFilter_txx
#define __itkCorrectGMUsingPVMapFilter_txx

#include "itkCorrectGMUsingPVMapFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "ConversionUtils.h"

#include "itkLogHelper.h"

namespace itk
{
template <typename TImageType >  
CorrectGMUsingPVMapFilter<TImageType>
::CorrectGMUsingPVMapFilter()
{
  m_DoGreyMatterCheck = true;
  m_DoCSFCheck = true;
  m_GreyMatterThreshold = 1;
  niftkitkDebugMacro(<< "CorrectGMUsingPVMapFilter():Constructed with m_DoGreyMatterCheck=" << m_DoGreyMatterCheck \
      << ", m_DoCSFCheck=" << m_DoCSFCheck \
      << ", m_GreyMatterThreshold=" << m_GreyMatterThreshold \
      );
}

template <typename TImageType >  
void 
CorrectGMUsingPVMapFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GreyMatterThreshold:" << m_GreyMatterThreshold << std::endl;
  os << indent << "DoGreyMatterCheck:" << m_DoGreyMatterCheck << std::endl;
  os << indent << "DoCSFCheck:" << m_DoCSFCheck << std::endl;
}

template <typename TImageType >  
void
CorrectGMUsingPVMapFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<< "GenerateData():Starting");
  
  typename InputImageType::Pointer inputSegmentedImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));       
  typename InputImageType::Pointer inputGMPVMap = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));       
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  this->CheckInputsAndOutputsSameSize();
  
  // Make sure that input segmented image only has 3 values.
  this->CheckOrAssignLabels();
    
  // Make sure that GMPVC map only contains values between 0 and 1.
  this->CheckPVMap("GMPVC", inputGMPVMap.GetPointer());
  
  // Now the main algorithm.
  this->AllocateOutputs();
  
  m_ListOfGreyMatterVoxelsBeforeCorrection.clear();
  m_ListOfGreyMatterVoxelsAfterCorrection.clear();
  
  InputPixelType segmentedValue;
  InputPixelType greyMatterPVMapValue;
  InputIndexType segmentedIndex;
  OutputPixelType outputValue;
  
  ImageRegionConstIteratorWithIndex<InputImageType> segmentedIterator(inputSegmentedImage, inputSegmentedImage->GetLargestPossibleRegion());
  ImageRegionConstIteratorWithIndex<InputImageType> greyMatterIterator(inputGMPVMap, inputGMPVMap->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  
  segmentedIterator.GoToBegin();
  greyMatterIterator.GoToBegin();
  outputIterator.GoToBegin();
  
  unsigned int removedDueToGreyMatterCheck = 0;
  unsigned int removedDueToCSFCheck = 0;
  
  while(!segmentedIterator.IsAtEnd() && !greyMatterIterator.IsAtEnd() && !outputIterator.IsAtEnd())
    {
      segmentedValue = segmentedIterator.Get();
      segmentedIndex = segmentedIterator.GetIndex();
      
      greyMatterPVMapValue = greyMatterIterator.Get();

      if (segmentedValue == this->GetGreyMatterLabel())
        {
          m_ListOfGreyMatterVoxelsBeforeCorrection.push_back(segmentedIterator.GetIndex());
        }

      // By default, we just pass through.
      outputValue = segmentedValue;  
      
      if (m_DoGreyMatterCheck)
        {
          if (segmentedValue == this->GetGreyMatterLabel() 
              && this->IsOnCSFBoundary(inputSegmentedImage, segmentedIndex, true) 
              && greyMatterPVMapValue < m_GreyMatterThreshold)
            {
              outputValue = this->GetExtraCerebralMatterLabel();
              removedDueToGreyMatterCheck++;
            }
          else if (segmentedValue == this->GetGreyMatterLabel()  
                   && this->IsOnWMBoundary(inputSegmentedImage, segmentedIndex, true) 
                   && greyMatterPVMapValue < m_GreyMatterThreshold)
            {
              outputValue = this->GetWhiteMatterLabel();
              removedDueToGreyMatterCheck++;
            }
        }
      
      if (m_DoCSFCheck)
        {
          if (segmentedValue == this->GetExtraCerebralMatterLabel()
              && this->IsOnWMBoundary(inputSegmentedImage, segmentedIndex, true))
            {
              outputValue = this->GetGreyMatterLabel();
              removedDueToCSFCheck++;
            }
          else if (segmentedValue == this->GetWhiteMatterLabel()
                   && this->IsOnCSFBoundary(inputSegmentedImage, segmentedIndex, true))
            {
              outputValue = this->GetGreyMatterLabel();
              removedDueToCSFCheck++;
            }
        }
      
      if (outputValue == this->GetGreyMatterLabel())
        {
          m_ListOfGreyMatterVoxelsAfterCorrection.push_back(segmentedIterator.GetIndex());  
        }

      outputIterator.Set(outputValue);

      ++segmentedIterator;
      ++greyMatterIterator;
      ++outputIterator;
    }
  
  niftkitkDebugMacro(<< "GenerateData():Finished, started with:" << m_ListOfGreyMatterVoxelsBeforeCorrection.size() \
      << ", and finished with:" << m_ListOfGreyMatterVoxelsAfterCorrection.size() \
      << ", removed " << removedDueToGreyMatterCheck \
      << " due to grey matter check and " << removedDueToCSFCheck \
      << " due to CSF check" \
      );
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
