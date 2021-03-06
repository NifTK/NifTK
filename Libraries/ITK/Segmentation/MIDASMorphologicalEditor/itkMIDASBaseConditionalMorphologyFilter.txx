/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASBaseConditionalMorphologyFilter_txx
#define itkMIDASBaseConditionalMorphologyFilter_txx

#include "itkMIDASBaseConditionalMorphologyFilter.h"
#include <itkMIDASHelper.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkConstNeighborhoodIterator.h>

namespace itk
{

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASBaseConditionalMorphologyFilter()
  {
    this->SetNumberOfRequiredInputs(1); // second input is optional
    this->SetNumberOfRequiredOutputs(1);
    m_InValue = 1;
    m_OutValue = 0;
    m_NumberOfIterations = 0;
    
    // This is a member variable, so we don't repeatedly create/destroy the memory if the main filter is called repeatedly.
    m_TempImage = MaskImageDuplicatorType::New();
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_InValue=" << m_InValue << std::endl;
    os << indent << "m_OutValue=" << m_OutValue << std::endl;
    os << indent << "m_NumberOfIterations=" << m_NumberOfIterations << std::endl;
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASBaseConditionalMorphologyFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetGreyScaleImageInput(const InputMainImageType *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< InputMainImageType * >( input ) );
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASBaseConditionalMorphologyFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetBinaryImageInput(const InputMaskImageType *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputMaskImageType * >( input ) );
                                     
    m_TempImage->SetInputImage(input);
    m_TempImage->Update();                        
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>
  ::CopyImageData(OutputImageType* in, OutputImageType *out)
  {
    ImageRegionConstIterator<OutputImageType> inIter(in, in->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType> outIter(out, out->GetLargestPossibleRegion());
    
    for (inIter.GoToBegin(), outIter.GoToBegin();
         !inIter.IsAtEnd(); // both images should always be same size, so we only check one of them
         ++inIter, ++outIter)
    {
      outIter.Set(inIter.Get());
    }
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  bool 
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>
  ::IsOnBoundaryOfImage(const OutputImageIndexType &voxelIndex, const OutputImageSizeType &size)
  {
    for (int i = 0; i < TInputImage1::ImageDimension; i++)
    {
      if((int)voxelIndex[i] == (int)0 || (int)voxelIndex[i] == (int)size[i]-1)
      {
        return true;
      }
    }
    return false;
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  bool
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage> 
  ::IsOnBoundaryOfRegion(const OutputImageIndexType &voxelIndex, const OutputImageRegionType& region)
  {
    for (int i = 0; i < TInputImage1::ImageDimension; i++)
    {
      if((int)voxelIndex[i] == (int)(region.GetIndex()[i]) 
      || (int)voxelIndex[i] == (int)(region.GetIndex()[i] + region.GetSize()[i]-1)
        )
      {
        return true;
      }
    }
    return false;
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>
  ::DoOneIterationOfFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out)
  {
    this->BeforeIteration();
    this->DoFilter(inGrey, inMask, out);
    this->AfterIteration();
  }

  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>::GenerateData()
  {    
    this->AllocateOutputs();

    // Check input binary mask is set.
    InputMaskImageType *inputMaskImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(0));
    if(!inputMaskImage)
    {
      itkExceptionMacro(<< "Input binary mask is not set!");
    }
    
    // Give subclasses chance to set things up before we call main filter body.
    this->BeforeFilter();
    
    // Get inputs and outputs once, before we start looping
    InputMaskImagePointer  inputMaskImagePtr = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(0));
    InputMainImagePointer  inputMainImagePtr = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(1));
    OutputImagePointer     outputImagePtr = this->GetOutput();
    
    // If we have zero iterations, we can copy the input mask to output, and exit.    
    if (m_NumberOfIterations == 0)
    {
      this->CopyImageData(inputMaskImagePtr, outputImagePtr);
      return;
    }

    // If we have 1 iteration, we don't require any temporary images, 
    // so we read from the input, and write straight to the output 
    if (m_NumberOfIterations == 1)
    {
      outputImagePtr->FillBuffer(this->GetOutValue());
      this->DoOneIterationOfFilter(inputMainImagePtr.GetPointer(), inputMaskImagePtr.GetPointer(), outputImagePtr.GetPointer());
      return;
    }

    // So, we have at least 2 iterations, so we must have an 
    // up to date (i.e. correct size) temporary buffer.
    
    if (m_TempImage->GetOutput()->GetLargestPossibleRegion().GetSize() != inputMaskImagePtr->GetLargestPossibleRegion().GetSize())
    {
      m_TempImage->Update();
    }
    
    OutputImageType *readImagePtr(NULL);
    OutputImageType *writeImagePtr(NULL);
    
    for(unsigned int iterationIndex = 0; iterationIndex < m_NumberOfIterations; iterationIndex++)
    {
      if (iterationIndex == 0)
      {
        readImagePtr = inputMaskImagePtr;
        writeImagePtr = m_TempImage->GetOutput();
      }
      else if (iterationIndex % 2 == 1)
      {
        readImagePtr = m_TempImage->GetOutput();
        writeImagePtr = outputImagePtr;
      }
      else if (iterationIndex % 2 == 0)
      {
        readImagePtr = outputImagePtr;
        writeImagePtr = m_TempImage->GetOutput();
      }
      
      this->DoOneIterationOfFilter(inputMainImagePtr.GetPointer(), readImagePtr, writeImagePtr);
    }

    // Data gets written, alternating between the output image, and the temporary buffer.
    // iterations == 0 (see above)
    // iterations == 1 : read from input, dilation/erosion writes straight to output
    // iterations == 2 : read from input, dilation/erosion 1 goes to temporary, dilation/erosion 2 goes to output
    // iterations == 3 : read from input, dilation/erosion 1 goes to temporary, dilation/erosion 2 goes to output, dilation/erosion 3 goes to temporary, therefore we need to copy back to output.
    // etc.
    
    if (m_NumberOfIterations % 2 == 1 && m_NumberOfIterations > 1)
    {
      CopyImageData(writeImagePtr, outputImagePtr);
    }

    // Give subclasses chance to tidy things up, or close things down before we exit this filter.
    this->AfterFilter();

  }

}

#endif
