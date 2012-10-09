/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASMeanIntensityWithinARegionFilter_txx
#define itkMIDASMeanIntensityWithinARegionFilter_txx

#include "itkMIDASMeanIntensityWithinARegionFilter.h"
#include "itkImageRegionConstIterator.h"

#include <map>
#include <algorithm>

namespace itk
{
  //Constructor
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASMeanIntensityWithinARegionFilter()
  {
    this->SetNumberOfRequiredOutputs(1);
    m_MeanIntensityMainImage = 0.0;
    m_InValue = 1;
    m_Counter = 0;
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_MeanIntensityMainImage=" << m_MeanIntensityMainImage << std::endl;
    os << indent << "m_Counter=" << m_Counter << std::endl;
    os << indent << "m_InValue=" << m_InValue << std::endl;
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::AllocateOutputs()
  {
    // Pass the main image input through as the output
    InputMaskImagePointer maskImagePtr = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    this->GraftOutput(maskImagePtr);
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::SetGreyScaleImageInput(const InputMainImageType* image)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputMainImageType * >( image ) );
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::SetBinaryImageInput(const InputMaskImageType* image)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< InputMaskImageType * >( image ) );
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::BeforeThreadedGenerateData()
  {   
    const unsigned int numberOfInputImages = this->GetNumberOfInputs(); 
    
    if(numberOfInputImages != 2)
    {
      itkExceptionMacro(<< "There should be exactly two input images for MIDASMeanIntensityWithinARegionFilter. ");
    }
    
    // Get the input and output pointers
    InputMainImageConstPointer  inputMainImgPtr = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    InputMaskImageConstPointer  inputMaskImgPtr = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    
    InputMainImageRegionType regionMain;
    for(unsigned int i = 0; i < numberOfInputImages; i++)
    {
      // Check each input is set.
      InputMainImageType *inputImage = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(i));
      if(!inputImage)
      {
        itkExceptionMacro(<< "Input " << i << " not set!");
      }
        
      // Check they are the same size.
      if(i == 0)
      {
        regionMain = inputImage->GetLargestPossibleRegion();
      }
      else if(i == 1)
      {
        if(regionMain != inputImage->GetLargestPossibleRegion())
        {
          itkExceptionMacro(<< "All Inputs must have the same dimensions.");
        }
      }
    } 
    
    unsigned int numberOfThreads = this->GetNumberOfThreads();
    m_TotalIntensityVector.reserve(numberOfThreads);
    m_CountPixelsVector.reserve(numberOfThreads);
    
    //just filling some dummy values
    for(unsigned int i = 0; i < numberOfThreads; i++)
    {
      m_TotalIntensityVector.push_back(0.0);
      m_CountPixelsVector.push_back(0);
    }
  
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>
    ::ThreadedGenerateData(const InputMainImageRegionType &outputRegionForThread, int ThreadID)
  {   
    // Get the input and output pointers
    InputMainImageConstPointer  inputMainImgPtr = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    InputMaskImageConstPointer  inputMaskImgPtr = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));

    // Define/declare an iterator that will walk the output region for this
    // thread.
    typedef ImageRegionConstIterator<InputMainImageType> InputIteratorMain;
    typedef ImageRegionConstIterator<InputMaskImageType> InputIteratorMask;
    
    InputIteratorMain  mainInputIter(inputMainImgPtr, outputRegionForThread);
    InputIteratorMask  maskInputIter(inputMaskImgPtr, outputRegionForThread);
            
    double sumIntensity = 0.0;
    unsigned long int countPixels = 0;

    mainInputIter.GoToBegin();
    maskInputIter.GoToBegin();
    
    // walk the regions, find out the mean intensity of each pixel
    while( !mainInputIter.IsAtEnd() && !maskInputIter.IsAtEnd() ) 
    {
      const InputMaskImagePixelType valueMask = maskInputIter.Get();
      if(valueMask == m_InValue)
      {
        const InputMainImagePixelType valueMain = mainInputIter.Get();
        sumIntensity += valueMain;
        ++countPixels;    
      }
    
      ++mainInputIter;
      ++maskInputIter;
    }

    m_TotalIntensityVector[ThreadID] = sumIntensity;
    m_CountPixelsVector[ThreadID]    = countPixels;

  }    


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::AfterThreadedGenerateData()
  {   
    double totalIntensity = 0.0;
    unsigned int numberOfThreads = this->GetNumberOfThreads();
    
    m_Counter = 0;
    
    for(unsigned int i = 0; i < numberOfThreads; i++)
    { 
      totalIntensity += m_TotalIntensityVector[i];
      m_Counter += m_CountPixelsVector[i];
    }
    
    m_MeanIntensityMainImage = totalIntensity / (double) m_Counter;
  }

  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  double MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::GetMeanIntensityMainImage()
  {
    return m_MeanIntensityMainImage;   
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  unsigned long int MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage>::GetCount()
  {
    return m_Counter;   
  }

}//end namespace itk


#endif
