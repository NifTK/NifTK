/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 13:19:18 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7816 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASDownSamplingFilter_txx
#define itkMIDASDownSamplingFilter_txx

#include "itkMIDASDownSamplingFilter.h"
#include "itkImageRegion.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLogHelper.h"

namespace itk
{
  //Constructor
  template <class TInputImage, class TOutputImage>
  MIDASDownSamplingFilter<TInputImage, TOutputImage>::MIDASDownSamplingFilter()
  {
    this->SetNumberOfRequiredOutputs(1);
    m_DownSamplingFactor = 1;
    m_InValue = 1;
    m_OutValue = 0;
  }


  template <class TInputImage, class TOutputImage>
  void MIDASDownSamplingFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_DownSamplingFactor=" << m_DownSamplingFactor << std::endl;
    os << indent << "m_InValue=" << m_InValue << std::endl;
    os << indent << "m_OutValue=" << m_OutValue << std::endl;
  }


  template <class TInputImage, class TOutputImage>
  void MIDASDownSamplingFilter<TInputImage, TOutputImage>::GenerateData()
  {   
    const unsigned int numberOfInputImages = this->GetNumberOfInputs(); 
    this->AllocateOutputs();
    
    if(numberOfInputImages != 1)
    {
      niftkitkErrorMacro(<< "There should be one input image for MIDASDownSamplingFilter. ");
    }
    
    //Check input image is set.
    InputImageType *inputImage = static_cast<InputImageType*>(this->ProcessObject::GetInput(0));
    if(!inputImage)
    {
      niftkitkErrorMacro(<< "Input image is not set!");
    }
    
    OutputImagePointer outputImagePtr = this->GetOutput();
    if (outputImagePtr.IsNull())
    {
      niftkitkErrorMacro(<< "Output image is not set!");
    }
    
    if(m_DownSamplingFactor <= 1)
    {
      niftkitkErrorMacro(<< "Down Sample factor is not valid. It should be a positive integer greater than 1!");
    }

    if (TInputImage::ImageDimension != 2 && TInputImage::ImageDimension != 3)
    {
      niftkitkErrorMacro(<< "Unsupported image dimension. This filter only does 2D or 3D images");
    }
    
    //Define an iterator that will walk the input image
    typedef ImageRegionIterator<InputImageType> InputImageIterator;
    InputImageIterator inputImageIter(inputImage, inputImage->GetLargestPossibleRegion());

    //Define an iterator that will walk the output image
    typedef ImageRegionIteratorWithIndex<OutputImageType> OutputImageIterator;
    OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

    //set all the pixel values of the output image to the out value first (normally zero).
    for(outputImageIter.GoToBegin(); !outputImageIter.IsAtEnd(); ++outputImageIter)
    {
      outputImageIter.Set(m_OutValue);  
    }
    
    InputImageSizeType inputImageSize = inputImage->GetLargestPossibleRegion().GetSize();
    InputImageIndexType inputImageIndex;
    OutputImageIndexType outputImageIndex;
    
    inputImageIndex.Fill(0);
    outputImageIndex.Fill(0);    
    inputImageIter.GoToBegin();
    
    if (TInputImage::ImageDimension == 3)
    {
      // Iterating through the input image, z dimension = index 2.
      for(inputImageIndex[2] = 0, outputImageIndex[2] = 0; inputImageIndex[2] < (int)inputImageSize[2]; inputImageIndex[2]++, outputImageIndex[2] += ((inputImageIndex[2] % m_DownSamplingFactor) == 0 ? 1 : 0))
      {
        // Iterating through the input image, y dimension = index 1.
        for(inputImageIndex[1] = 0, outputImageIndex[1] = 0; inputImageIndex[1] < (int)inputImageSize[1]; inputImageIndex[1]++, outputImageIndex[1] += ((inputImageIndex[1] % m_DownSamplingFactor) == 0 ? 1 : 0))
        {
          // Iterating through the input image, x dimension = index 0.
          for(inputImageIndex[0] = 0, outputImageIndex[0] = 0; inputImageIndex[0] < (int)inputImageSize[0]; inputImageIndex[0]++, outputImageIndex[0] += ((inputImageIndex[0] % m_DownSamplingFactor) == 0 ? 1 : 0))
          {
            outputImageIter.SetIndex(outputImageIndex);
            
            if (inputImageIter.Get() != m_OutValue)
            {
              outputImageIter.Set(m_InValue);
            }
            ++inputImageIter;
          }
        }
      }
    }
    // If not 3D, must be 2D.
    else
    {
      // Iterating through the input image, y dimension = index 1.
      for(inputImageIndex[1] = 0, outputImageIndex[1] = 0; inputImageIndex[1] < (int)inputImageSize[1]; inputImageIndex[1]++, outputImageIndex[1] += ((inputImageIndex[1] % m_DownSamplingFactor) == 0 ? 1 : 0))
      {
        // Iterating through the input image, x dimension = index 0.
        for(inputImageIndex[0] = 0, outputImageIndex[0] = 0; inputImageIndex[0] < (int)inputImageSize[0]; inputImageIndex[0]++, outputImageIndex[0] += ((inputImageIndex[0] % m_DownSamplingFactor) == 0 ? 1 : 0))
        {
          outputImageIter.SetIndex(outputImageIndex);
          
          if (inputImageIter.Get() != m_OutValue)
          {
            outputImageIter.Set(m_InValue);
          }
          ++inputImageIter;
        }
      }
    }
        
  }//end of generatedata method


  template <class TInputImage, class TOutputImage>
  void MIDASDownSamplingFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
  {  
    //Call the superclass implementation of this method
    SuperClass::GenerateInputRequestedRegion();
    
    //Get pointers to the input and output
    InputImagePointer inputImagePtr   = const_cast<TInputImage *> (this->GetInput());
    OutputImagePointer outputImagePtr = this->GetOutput();
    
    if(!inputImagePtr)
    {
      niftkitkInfoMacro(<< "inputImagePtr is NULL");
    }

    if(!outputImagePtr)
    {
      niftkitkInfoMacro(<< "outputImagePtr is NULL");
    }
   
    typename TInputImage::RegionType inputImageRequestedRegion = inputImagePtr->GetLargestPossibleRegion();
    inputImagePtr->SetRequestedRegion(inputImageRequestedRegion);
    
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputImageRequestedRegion.GetSize()=" << inputImageRequestedRegion.GetSize());
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputImageRequestedRegion.GetIndex()=" << inputImageRequestedRegion.GetIndex());
    
  }


  template <class TInputImage, class TOutputImage>
  void MIDASDownSamplingFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
  {  
    //Call the supreclass implementation of this method
    SuperClass::GenerateOutputInformation();
    
    //Get pointers to the input and output
    InputImagePointer inputImagePtr   = const_cast<TInputImage *> (this->GetInput());
    OutputImagePointer outputImagePtr = this->GetOutput();
    
    if(!inputImagePtr)
    {
      niftkitkInfoMacro(<< "inputImagePtr is NULL");
    }

    if(!outputImagePtr)
    {
      niftkitkInfoMacro(<< "outputImagePtr is NULL");
    }

    const typename TInputImage::PointType     inputImageOrigin     = inputImagePtr->GetOrigin();
    const typename TInputImage::SpacingType   inputImageSpacing    = inputImagePtr->GetSpacing();   
    const typename TInputImage::DirectionType inputImageDirection  = inputImagePtr->GetDirection();
    const typename TInputImage::SizeType      inputImageSize       = inputImagePtr->GetLargestPossibleRegion().GetSize(); 
    const typename TInputImage::IndexType     inputImageStartIndex = inputImagePtr->GetLargestPossibleRegion().GetIndex();

    niftkitkDebugMacro(<< "GenerateOutputInformation():Input size=" << inputImageSize << ", index=" << inputImageStartIndex << ", spacing=" << inputImageSpacing << ", origin=" << inputImageOrigin);
    
    typename TOutputImage::PointType     outputImageOrigin;
    typename TOutputImage::SpacingType   outputImageSpacing;
    typename TOutputImage::DirectionType outputImageDirection;
    typename TOutputImage::SizeType      outputImageSize;
    typename TOutputImage::IndexType     outputImageIndex;
    
    for(unsigned int i = 0; i < TInputImage::ImageDimension; i++)
    {
      // From MIDAS code, https://cmicdev.cs.ucl.ac.uk/trac/ticket/766
      
      outputImageSize[i] = (unsigned long) vcl_floor(((inputImageSize[i] - 1)/ (double)m_DownSamplingFactor ) + 1);
      if(outputImageSize[i] < 1)
      {
        outputImageSize[i] = 1;
      }

      // Adjust spacing to cover the same dimension
      outputImageSpacing[i] = (inputImageSpacing[i] * inputImageSize[i]) / (double) outputImageSize[i];

      // Adjust origin to make the images perfectly registered.
      outputImageOrigin[i] = inputImageOrigin[i]
                             + ((inputImageSize[i]-1)/2.0)*inputImageSpacing[i]   // Distance to middle on input image
                             - ((outputImageSize[i]-1)/2.0)*outputImageSpacing[i] // Distance back to corner on output image
                             ;
      
      // Making the index identical, as we shall make sure we always process the whole image, so index is always [0,0,0]
      outputImageIndex[i] = inputImageStartIndex[i];
    }
    outputImageDirection = inputImageDirection;

    niftkitkDebugMacro(<< "GenerateOutputInformation():Output size=" << outputImageSize << ", index=" << outputImageIndex << ", spacing=" << outputImageSpacing << ", origin=" << outputImageOrigin);
       
    outputImagePtr->SetDirection(outputImageDirection);
    outputImagePtr->SetOrigin(outputImageOrigin);
    outputImagePtr->SetSpacing(outputImageSpacing);

    typename TOutputImage::RegionType outputImageRegion;
    outputImageRegion.SetSize(outputImageSize);
    outputImageRegion.SetIndex(outputImageIndex);
    outputImagePtr->SetRegions(outputImageRegion);    
  }


}//end namespace itk


#endif
