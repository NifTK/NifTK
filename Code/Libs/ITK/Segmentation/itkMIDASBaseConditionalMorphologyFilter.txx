/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASBaseConditionalMorphologyFilter_txx
#define itkMIDASBaseConditionalMorphologyFilter_txx

#include "itkMIDASBaseConditionalMorphologyFilter.h"
#include "itkMIDASHelper.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkConstNeighborhoodIterator.h"

namespace itk
{

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASBaseConditionalMorphologyFilter()
  {
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
    this->ProcessObject::SetNthInput(0, const_cast< InputMainImageType * >( input ) );
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASBaseConditionalMorphologyFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetBinaryImageInput(const InputMaskImageType *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< InputMaskImageType * >( input ) );
                                     
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
  ::IsOnBoundaryOfImage(OutputImageIndexType &voxelIndex, OutputImageSizeType &size)
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
    const unsigned int numberOfInputImages = this->GetNumberOfInputs(); 
    this->AllocateOutputs();
    
    if(numberOfInputImages != 2)
    {
      itkExceptionMacro(<< "There should be two input images for subclasses of MIDASBaseConditionalMorphologyFilter. ");
    }
    
    //Check input image is set.
    InputMainImageType *inputMainImage = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    if(!inputMainImage)
    {
      itkExceptionMacro(<< "Input greyscale image is not set!");
    }
    
    //Check input binary mask is set.
    InputMaskImageType *inputMaskImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    if(!inputMaskImage)
    {
      itkExceptionMacro(<< "Input binary mask is not set!");
    }

    if( (inputMainImage->GetLargestPossibleRegion().GetSize()) != (inputMaskImage->GetLargestPossibleRegion().GetSize()) )
    { 
      itkExceptionMacro(<< "Input greyscale and binary images don't match in size!");
    }
    
    // Give subclasses chance to set things up before we call main filter body.
    this->BeforeFilter();
    
    // Get inputs and outputs once, before we start looping
    InputMainImagePointer  inputMainImagePtr = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    InputMaskImagePointer  inputMaskImagePtr = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    OutputImagePointer     outputImagePtr = this->GetOutput();
    
    // If we have zero iterations, we can copy the input mask to output, and exit.    
    if (m_NumberOfIterations == 0)
    {
      this->CopyImageData(inputMaskImagePtr, outputImagePtr);
      return;
    }
    
    // If we have 1 iteration, we don't require any temporary images, 
    // so we read from the input, and write straight to the output 
    else if (m_NumberOfIterations == 1)
    {
      this->DoOneIterationOfFilter(inputMainImagePtr.GetPointer(), inputMaskImagePtr.GetPointer(), outputImagePtr.GetPointer());
      return;
    }

    // So, we have at least 2 iterations, so we must have an 
    // up to date (i.e. correct size) temporary buffer.
    
    if (m_TempImage->GetOutput()->GetLargestPossibleRegion().GetSize() != inputMaskImagePtr->GetLargestPossibleRegion().GetSize())
    {
      m_TempImage->Update();
    }
    
    OutputImageType *readImagePtr;
    OutputImageType *writeImagePtr;
    
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

  }//end of generatedata method

}//end namespace itk


#endif
