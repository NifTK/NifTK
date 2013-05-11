/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASMaskByRegionImageFilter_txx
#define itkMIDASMaskByRegionImageFilter_txx

#include "itkMIDASMaskByRegionImageFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageFileWriter.h>
#include <itkUCLMacro.h>
#include <itkLogHelper.h>

namespace itk
{

  template <class TInputImage, class TOutputImage>
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>::MIDASMaskByRegionImageFilter()
  {
    IndexType dummyIndex; dummyIndex.Fill(0);
    SizeType dummySize; dummySize.Fill(0);
    m_Region.SetSize(dummySize);
    m_Region.SetIndex(dummyIndex);
    m_OutputBackgroundValue = 0;
    m_UserSetRegion = false;
  }

  template <class TInputImage, class TOutputImage>
  void 
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_Region=" << m_Region << std::endl;
    os << indent << "m_OutputBackgroundValue=" << m_OutputBackgroundValue << std::endl;
    os << indent << "m_UserSetRegion=" << m_UserSetRegion << std::endl;
  }

  template <class TInputImage, class TOutputImage>
  void
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::BeforeThreadedGenerateData()
  {
    // Get the input and output pointers, check same size image
    typename TInputImage::Pointer inputPtr = static_cast<TInputImage*>(this->ProcessObject::GetInput(0));
    typename TOutputImage::Pointer outputPtr = static_cast<TOutputImage*>(this->ProcessObject::GetOutput(0));

    if (inputPtr->GetLargestPossibleRegion().GetSize() != outputPtr->GetLargestPossibleRegion().GetSize())
    {
      niftkitkDebugMacro(<< "Input 0 and output are not the same size??? They should be.");
    }
    
    typename TInputImage::Pointer inputPtr1 = static_cast<TInputImage*>(this->ProcessObject::GetInput(1));
    if (inputPtr1.IsNotNull())
    {
      if (inputPtr1->GetLargestPossibleRegion().GetSize() != outputPtr->GetLargestPossibleRegion().GetSize())
      {
        niftkitkDebugMacro(<< "Input 1 and output are not the same size??? They should be.");
      }
    }

    typename TInputImage::Pointer inputPtr2 = static_cast<TInputImage*>(this->ProcessObject::GetInput(2));
    if (inputPtr2.IsNotNull())
    {
      if (inputPtr2->GetLargestPossibleRegion().GetSize() != outputPtr->GetLargestPossibleRegion().GetSize())
      {
        niftkitkDebugMacro(<< "Input 2 and output are not the same size??? They should be.");
      }
    }
    
    // Fill output buffer with background value.
    this->GetOutput()->FillBuffer(m_OutputBackgroundValue);
  }

  template <class TInputImage, class TOutputImage>
  void
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::ThreadedGenerateData(const RegionType& outputRegionForThread, int threadNumber) 
  {
    // Get the input and output pointers
    typename TInputImage::Pointer inputPtr = static_cast<TInputImage*>(this->ProcessObject::GetInput(0));
    typename TOutputImage::Pointer outputPtr = static_cast<TOutputImage*>(this->ProcessObject::GetOutput(0));
    
    // We pre-calculate (i.e. before main loop), the correct region.
    RegionType actualRegion = outputRegionForThread;
    IndexType actualIndex = actualRegion.GetIndex();
    SizeType actualSize = actualRegion.GetSize();
        
    if (m_UserSetRegion)
    {
      IndexType maxIndexOutput;
      IndexType maxIndexUser;
      for (int i = 0; i < TInputImage::ImageDimension; i++)
      {
        maxIndexOutput[i] = outputRegionForThread.GetIndex()[i] + outputRegionForThread.GetSize()[i] - 1;
        maxIndexUser[i] = m_Region.GetIndex()[i] + m_Region.GetSize()[i] - 1;
      }
      // The whole region is split up according to thread,
      // and the user may specify a restricted region.
      // We need the intersection of the two.
      
      for (int i = 0; i < TInputImage::ImageDimension; i++)
      {
        actualIndex[i] = std::max(outputRegionForThread.GetIndex()[i], m_Region.GetIndex()[i]);
        actualSize[i] = std::min(maxIndexUser[i], maxIndexOutput[i]) - actualIndex[i] + 1;
      }
      actualRegion.SetSize(actualSize);
      actualRegion.SetIndex(actualIndex);
    }
    
    ImageRegionConstIteratorWithIndex<TInputImage> inputIt(inputPtr, actualRegion);
    ImageRegionIterator<TOutputImage> outputIt(outputPtr, actualRegion);
     

    // If input 1 and input 2 are specified, we are additionally using 2 input images to mask.    
    typename TInputImage::Pointer additionsImage = dynamic_cast<TInputImage*>(this->ProcessObject::GetInput(1));
    typename TInputImage::Pointer connectionBreakerImage = dynamic_cast<TInputImage*>(this->ProcessObject::GetInput(2));
    
    if (additionsImage.IsNotNull() && connectionBreakerImage.IsNotNull())
    {
      InputPixelType  inputPixelValue;
      InputPixelType  additionsPixelValue;
      InputPixelType  subtractionsPixelValue;
      OutputPixelType outputPixelValue;
      ImageRegionConstIteratorWithIndex<TInputImage> additionsImageIterator(additionsImage, actualRegion);
      ImageRegionConstIteratorWithIndex<TInputImage> connectionBreakerImageIterator(connectionBreakerImage, actualRegion);
      
      for (inputIt.GoToBegin(),
           additionsImageIterator.GoToBegin(),
           connectionBreakerImageIterator.GoToBegin(),
           outputIt.GoToBegin(); 
           !inputIt.IsAtEnd(); 
           ++inputIt,
           ++additionsImageIterator,
           ++connectionBreakerImageIterator, 
           ++outputIt)
      {
        inputPixelValue = inputIt.Get();
        additionsPixelValue = additionsImageIterator.Get();
        subtractionsPixelValue = connectionBreakerImageIterator.Get();
        
        // See header file for spec.
        if ((inputPixelValue != 0 || additionsPixelValue != 0) &&  subtractionsPixelValue == 0)
        {
          outputPixelValue = 1;
        }            
        else
        {
          outputPixelValue = 0;
        }        
        outputIt.Set(outputPixelValue);
      }
    } 
    else
    {
      for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt)
      {
        outputIt.Set(inputIt.Get());
      }
    }
  }
} // end namespace

#endif
