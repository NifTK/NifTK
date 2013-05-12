/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASUpSamplingFilter_txx
#define itkMIDASUpSamplingFilter_txx

#include "itkMIDASUpSamplingFilter.h"
#include <itkImageRegion.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLogHelper.h>

namespace itk
{

  //Constructor
  template <class TInputImage, class TOutputImage>
  MIDASUpSamplingFilter<TInputImage, TOutputImage>::MIDASUpSamplingFilter()
  {
    this->SetNumberOfRequiredOutputs(1);
    m_UpSamplingFactor = 1;
    m_InValue = 1;
    m_OutValue = 0;
  }


  template <class TInputImage, class TOutputImage>
  void MIDASUpSamplingFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_UpSamplingFactor=" << m_UpSamplingFactor << std::endl;
    os << indent << "m_InValue=" << m_InValue << std::endl;
    os << indent << "m_OutValue=" << m_OutValue << std::endl;
  }


  template <class TInputImage, class TOutputImage>
  void MIDASUpSamplingFilter<TInputImage, TOutputImage>::GenerateData()
  {   
    const unsigned int numberOfInputImages = this->GetNumberOfInputs(); 
    this->AllocateOutputs();
    
    if(numberOfInputImages != 2)
    {
      niftkitkDebugMacro(<< "There should be two input images for MIDASUpSamplingFilter. ");
    }
    
    // Check input image is set.
    InputImageType *inputImage = static_cast<InputImageType*>(this->ProcessObject::GetInput(0));
    if(!inputImage)
    {
      niftkitkDebugMacro(<< "Input downsized image is not set!");
    }

    OutputImagePointer outputImagePtr = this->GetOutput();
    if (outputImagePtr.IsNull())
    {
      niftkitkDebugMacro(<< "Output image is not set!");
    }
    
    if(m_UpSamplingFactor <= 1)
    {
      niftkitkDebugMacro(<< "Up Sample factor is not valid. It should be a positive integer greater than 1!");
    }

    if (TInputImage::ImageDimension != 2 && TInputImage::ImageDimension != 3)
    {
      niftkitkDebugMacro(<< "Unsupported image dimension. This filter only does 2D or 3D images");
    }

    // Define an iterator that will walk the down sized input image.
    typedef ImageRegionIteratorWithIndex<InputImageType> InputImageIterator;
    InputImageIterator inputImageIter(inputImage, inputImage->GetLargestPossibleRegion());

    // Define an iterator that will walk the output image
    typedef ImageRegionIterator<OutputImageType> OutputImageIterator;
    OutputImageIterator outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());

    outputImagePtr->FillBuffer(m_OutValue);

    OutputImageSizeType outputImageSize = outputImagePtr->GetLargestPossibleRegion().GetSize();
    InputImageIndexType inputImageIndex;
    OutputImageIndexType outputImageIndex;
    
    inputImageIndex.Fill(0);
    outputImageIndex.Fill(0);    
    outputImageIter.GoToBegin();
   
    if (TInputImage::ImageDimension == 3)
    {
      // Iterating through the input image, z dimension = index 2.
      for(inputImageIndex[2] = 0, outputImageIndex[2] = 0; outputImageIndex[2] < (int)outputImageSize[2]; outputImageIndex[2]++, inputImageIndex[2] += ((outputImageIndex[2] % m_UpSamplingFactor) == 0 ? 1 : 0))
      {
        // Iterating through the input image, y dimension = index 1.
        for(inputImageIndex[1] = 0, outputImageIndex[1] = 0; outputImageIndex[1] < (int)outputImageSize[1]; outputImageIndex[1]++, inputImageIndex[1] += ((outputImageIndex[1] % m_UpSamplingFactor) == 0 ? 1 : 0))
        {
          // Iterating through the input image, x dimension = index 0.
          for(inputImageIndex[0] = 0, outputImageIndex[0] = 0; outputImageIndex[0] < (int)outputImageSize[0]; outputImageIndex[0]++, inputImageIndex[0] += ((outputImageIndex[0] % m_UpSamplingFactor) == 0 ? 1 : 0))
          {
            inputImageIter.SetIndex(inputImageIndex);
            
            if (inputImageIter.Get() != m_OutValue)
            {
              outputImageIter.Set(m_InValue);
            }
            ++outputImageIter;
          }
        }
      }
    }
    // If not 3D, must be 2D.
    else
    {
      // Iterating through the input image, y dimension = index 1.
      for(inputImageIndex[1] = 0, outputImageIndex[1] = 0; outputImageIndex[1] < (int)outputImageSize[1]; outputImageIndex[1]++, inputImageIndex[1] += ((outputImageIndex[1] % m_UpSamplingFactor) == 0 ? 1 : 0))
      {
        // Iterating through the input image, x dimension = index 0.
        for(inputImageIndex[0] = 0, outputImageIndex[0] = 0; outputImageIndex[0] < (int)outputImageSize[0]; outputImageIndex[0]++, inputImageIndex[0] += ((outputImageIndex[0] % m_UpSamplingFactor) == 0 ? 1 : 0))
        {
          inputImageIter.SetIndex(inputImageIndex);
          
          if (inputImageIter.Get() != m_OutValue)
          {
            outputImageIter.Set(m_InValue);
          }
          ++outputImageIter;
        }
      }
    }
  }//end of generatedata method
  

  template <class TInputImage, class TOutputImage>
  void MIDASUpSamplingFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
  {  
    //Call the superclass implementation of this method
    SuperClass::GenerateInputRequestedRegion();
    
    // Get pointers to the input and output
    InputImagePointer inputDownSizedImagePtr = const_cast<TInputImage *> (this->GetInput(0));
    InputImagePointer inputFullSizedImagePtr = const_cast<TInputImage *> (this->GetInput(1));
    OutputImagePointer outputImagePtr = this->GetOutput();
    
    if(!inputFullSizedImagePtr)
    {
      niftkitkDebugMacro(<< "inputFullSizedImagePtr is NULL");
    }

    if(!inputDownSizedImagePtr)
    {
      niftkitkDebugMacro(<< "inputDownSizedImagePtr is NULL");
    }

    if(!outputImagePtr)
    {
      niftkitkDebugMacro(<< "outputImagePtr is NULL");
    }
   
    typename TInputImage::RegionType inputDownSizeRequestedRegion = inputDownSizedImagePtr->GetLargestPossibleRegion();
    inputDownSizedImagePtr->SetRequestedRegion(inputDownSizeRequestedRegion);

    typename TInputImage::RegionType inputFullSizeRequestedRegion = inputFullSizedImagePtr->GetLargestPossibleRegion();
    inputFullSizedImagePtr->SetRequestedRegion(inputFullSizeRequestedRegion);
    
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputDownSizeRequestedRegion.GetSize()=" << inputDownSizeRequestedRegion.GetSize());
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputDownSizeRequestedRegion.GetIndex()=" << inputDownSizeRequestedRegion.GetIndex());
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputFullSizeRequestedRegion.GetSize()=" << inputFullSizeRequestedRegion.GetSize());
    niftkitkDebugMacro(<< "GenerateInputRequestedRegion():inputFullSizeRequestedRegion.GetIndex()=" << inputFullSizeRequestedRegion.GetIndex());
    
  }


  template <class TInputImage, class TOutputImage>
  void MIDASUpSamplingFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
  {  
    // Call the supreclass implementation of this method
    SuperClass::GenerateOutputInformation();
    
    // Get pointers to the input and output
    InputImagePointer inputDownSizedImagePtr = const_cast<TInputImage *> (this->GetInput(0));
    InputImagePointer inputFullSizedImagePtr = const_cast<TInputImage *> (this->GetInput(1));
    OutputImagePointer outputImagePtr = this->GetOutput();
    
    if(!inputFullSizedImagePtr)
    {
      niftkitkDebugMacro(<< "inputFullSizedImagePtr is NULL");
    }

    if(!inputDownSizedImagePtr)
    {
      niftkitkDebugMacro(<< "inputDownSizedImagePtr is NULL");
    }

    if(!outputImagePtr)
    {
      niftkitkDebugMacro(<< "outputImagePtr is NULL");
    }
   
    const typename TInputImage::DirectionType inputImageDirection = inputFullSizedImagePtr->GetDirection();
    const typename TInputImage::PointType     inputImageOrigin    = inputFullSizedImagePtr->GetOrigin();
    const typename TInputImage::SpacingType   inputImageSpacing   = inputFullSizedImagePtr->GetSpacing();
    const typename TInputImage::SizeType      inputImageSize      = inputFullSizedImagePtr->GetLargestPossibleRegion().GetSize(); 
    const typename TInputImage::IndexType     inputImageIndex     = inputFullSizedImagePtr->GetLargestPossibleRegion().GetIndex();

    niftkitkDebugMacro(<< "GenerateOutputInformation():Input size=" << inputImageSize << ", index=" << inputImageIndex << ", spacing=" << inputImageSpacing << ", origin=" << inputImageOrigin);

    typename TOutputImage::PointType     outputImageOrigin = inputImageOrigin;
    typename TOutputImage::SpacingType   outputImageSpacing = inputImageSpacing;
    typename TOutputImage::DirectionType outputImageDirection = inputImageDirection;
    typename TOutputImage::SizeType      outputImageSize = inputImageSize;
    typename TOutputImage::IndexType     outputImageIndex = inputImageIndex;

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
