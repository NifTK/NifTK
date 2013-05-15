/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBaseCTEFilter_txx
#define __itkBaseCTEFilter_txx

#include "itkBaseCTEFilter.h"
#include <ConversionUtils.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include <itkLogHelper.h>

namespace itk
{
template <typename TImageType >  
BaseCTEFilter<TImageType>
::BaseCTEFilter()
{
  m_GreyMatterLabel = -1;
  m_WhiteMatterLabel = -1;
  m_ExtraCerebralMatterLabel = -1;
  m_UserHasSetTheLabelThresholds = false;

  niftkitkDebugMacro(<< "BaseCTEFilter():Constructed with" \
      << " GreyMatterLabel=" << m_GreyMatterLabel \
      << ", WhiteMatterLabel=" << m_WhiteMatterLabel \
      << ", ExtraCerebralMatterLabel=" << m_ExtraCerebralMatterLabel \
      << ", UserHasSetTheLabelThresholds=" << m_UserHasSetTheLabelThresholds);
}

template <typename TImageType >  
void 
BaseCTEFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GreyMatterLabel:" << m_GreyMatterLabel << std::endl;
  os << indent << "WhiteMatterLabel:" << m_WhiteMatterLabel << std::endl;
  os << indent << "ExtraCerebralMatterLabel:" << m_ExtraCerebralMatterLabel << std::endl;
  os << indent << "UserHasSetTheLabelThresholds:" << m_UserHasSetTheLabelThresholds << std::endl;
}

template <typename TImageType >
void
BaseCTEFilter<TImageType>
::CheckInputsAndOutputsSameSize()
{
  InputImageType *inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  InputSizeType inputSize = inputImage->GetLargestPossibleRegion().GetSize();
  
  for (unsigned int i = 1; i < this->GetNumberOfInputs(); i++)
    {
      inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(i));
      if (inputImage->GetLargestPossibleRegion().GetSize() != inputSize)
        {
          itkExceptionMacro(<< "Input image[" << i << "] size doesn't match size:" << inputSize);
        }
    }
  
  OutputImageType *output =  static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  OutputSizeType outputSize = output->GetLargestPossibleRegion().GetSize();
  
  if (outputSize != inputSize)
    {
      itkExceptionMacro(<< "Output image size:" << outputSize << ", doesn't match size:" << inputSize);  
    }

  return;
}

template <typename TImageType >  
void
BaseCTEFilter<TImageType>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // We need the LargestPossibleRegion, from all the inputs.
  InputImagePointer input;

  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++)
    {
      input = const_cast<InputImageType *>(this->GetInput(i));
      if ( input )
        {
    	  niftkitkDebugMacro(<<"GenerateInputRequestedRegion():Forcing input[" << i << "] to largest possible region:" << input->GetLargestPossibleRegion().GetSize());
          input->SetRequestedRegion( input->GetLargestPossibleRegion() );      
        }
      else
        {
    	  niftkitkInfoMacro(<<"GenerateInputRequestedRegion():Input[" << i << "] is present, but I couldn't request the region????");
        }
    }
}

template <typename TImageType >  
void
BaseCTEFilter<TImageType>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  output->SetRequestedRegionToLargestPossibleRegion();  
}

template <typename TImageType > 
void 
BaseCTEFilter<TImageType>
::SetLabelThresholds(InputPixelType greyMatter,
                     InputPixelType whiteMatter,
                     InputPixelType extraCerebralMatter)
{
  this->m_GreyMatterLabel = greyMatter;
  this->m_WhiteMatterLabel = whiteMatter;
  this->m_ExtraCerebralMatterLabel = extraCerebralMatter;
  this->m_UserHasSetTheLabelThresholds = true;

  niftkitkDebugMacro(<<"SetLabelThresholds():" \
      << " GreyMatterLabel=" << m_GreyMatterLabel \
      << ", WhiteMatterLabel=" << m_WhiteMatterLabel \
      << ", ExtraCerebralMatterLabel=" << m_ExtraCerebralMatterLabel \
      << ", UserHasSetTheLabelThresholds=" << m_UserHasSetTheLabelThresholds);
}                          
     
template <typename TImageType > 
bool 
BaseCTEFilter<TImageType>
::IsOnBoundary(const InputImageType *image, const InputIndexType& index, const InputPixelType boundaryValue, bool useFullyConnected)
{
  bool result = false;
  
  // We first check size, so that we don't accidentally run off the edge of the image.
  InputSizeType size = image->GetLargestPossibleRegion().GetSize();

  bool withinRange = true;
  for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
    {
      if (index[i] == (int)0 || index[i] == (int)(size[i] - 1))
        {
          withinRange = false;
        }
    } 
  
  if (withinRange)
    {
      if (useFullyConnected)
        {
          typename InputImageType::RegionType region;
          typename InputImageType::SizeType size;
          typename InputImageType::IndexType startingIndex;
          
          size.Fill(3);
          
          for (unsigned int i = 0; i < InputImageType::ImageDimension; i++)
            {
              startingIndex[i] = index[i] - 1;
            } 
          
          region.SetSize(size);
          region.SetIndex(startingIndex);
          
          ImageRegionConstIteratorWithIndex<InputImageType> neighbourHoodIterator(image, region);
          for (neighbourHoodIterator.GoToBegin(); !neighbourHoodIterator.IsAtEnd(); ++neighbourHoodIterator)
            {
              if (neighbourHoodIterator.Get() == boundaryValue)
                {
                  result = true;
                  break;
                }
            }          
        }
      else
        {
          InputIndexType indexPlus;
          InputIndexType indexMinus;
          
          for (unsigned int i = 0; i < Dimension; i++)
            {
              indexPlus = index;
              indexMinus = index;
              
              indexPlus[i] += 1;
              indexMinus[i] -= 1;

              if (image->GetPixel(indexPlus) == boundaryValue || image->GetPixel(indexMinus) == boundaryValue)
                {
                  result = true;  
                  break;
                }
            }            
        }      
    }
  return result;
}

} // end namespace

#endif
