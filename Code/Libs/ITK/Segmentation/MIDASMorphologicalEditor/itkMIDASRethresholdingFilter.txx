/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASRethresholdingFilter_txx
#define itkMIDASRethresholdingFilter_txx

#include "itkMIDASRethresholdingFilter.h"
#include "itkImageFileWriter.h"
#include "itkMIDASHelper.h"
#include "itkLogHelper.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASRethresholdingFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASRethresholdingFilter()
  {
    m_DownSamplingFactor = 1;
    m_InValue = 1;
    m_OutValue = 0;
    m_LowPercentageThreshold = 50;
    m_HighPercentageThreshold = 150;
    m_ROIImageFilter = ROIImageFilterType::New();
    m_DownSamplingFilter = DownSamplingFilterType::New();
    m_ErosionFilter1 = ErosionFilterType::New();
    m_ErosionFilter2 = ErosionFilterType::New();
    m_UpSamplingFilter = UpSamplingFilterType::New();
    m_PasteImageFilter = PasteImageFilterType::New();
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASRethresholdingFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_DownSamplingFactor=" << m_DownSamplingFactor << std::endl;
    os << indent << "m_InValue=" << m_InValue << std::endl;
    os << indent << "m_OutValue=" << m_OutValue << std::endl;
    os << indent << "m_LowPercentageThreshold=" << m_LowPercentageThreshold << std::endl;
    os << indent << "m_HighPercentageThreshold=" << m_HighPercentageThreshold << std::endl;
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASRethresholdingFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetGreyScaleImageInput(const InputMainImageType *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputMainImageType * >( input ) );
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASRethresholdingFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetBinaryImageInput(const InputMaskImageType *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< InputMaskImageType * >( input ) );
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void
  MIDASRethresholdingFilter<TInputImage1,TInputImage2, TOutputImage> 
  ::SetThresholdedImageInput(const InputMaskImageType* image)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(2, const_cast< InputMaskImageType * >( image ) );  
  }

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void
  MIDASRethresholdingFilter<TInputImage1,TInputImage2, TOutputImage> 
  ::CopyImageToOutput(OutputImageType* image)
  {
    // Copies input to output.
    OutputImagePointer outputImagePtr = this->GetOutput();
    
    // Check the sizes match.
    if( (image->GetLargestPossibleRegion().GetSize()) != (outputImagePtr->GetLargestPossibleRegion().GetSize()) )
    { 
      niftkitkDebugMacro(<< "Pipeline and output image don't match in size! You must always upate the LargestPossibleRegion.");
    }
    
    ImageRegionConstIterator<OutputImageType> inIter(image, image->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType> outIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());
    
    for (inIter.GoToBegin(), outIter.GoToBegin();
         !inIter.IsAtEnd(); // both images should always be same size, so we only check one of them
         ++inIter, ++outIter)
    {
      outIter.Set(inIter.Get());
    }
  }
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASRethresholdingFilter<TInputImage1,TInputImage2, TOutputImage>
  ::GenerateData()
  {
    const unsigned int numberOfInputImages = this->GetNumberOfInputs();    
     
    if(numberOfInputImages != 3)
    {
      itkExceptionMacro(<< "There should be three input images for MIDASRethresholdingFilter. ");
    }
    
    // Check input image is set.
    InputMainImageType *inputMainImage = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    if(!inputMainImage)
    {
      itkExceptionMacro(<< "Input image is not set!");
    }
    
    // Check input binary mask is set.
    InputMaskImageType *inputMaskImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    if(!inputMaskImage)
    {
      itkExceptionMacro(<< "Input mask is not set!");
    }

    // Check input thresholded mask is set.
    InputMaskImageType *inputThresholdImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(2));
    if(!inputThresholdImage)
    {
      itkExceptionMacro(<< "Input threshold image is not set!");
    }

    // Check the sizes match.
    if(    (inputMainImage->GetLargestPossibleRegion().GetSize()) != (inputMaskImage->GetLargestPossibleRegion().GetSize())
        || (inputMainImage->GetLargestPossibleRegion().GetSize()) != (inputThresholdImage->GetLargestPossibleRegion().GetSize())
      )
    { 
      itkExceptionMacro(<< "Input images don't match in size!");
    }
  
    this->AllocateOutputs();
    
    if (m_DownSamplingFactor <= 1)
    {
      this->CopyImageToOutput(inputMaskImage);
      return;
    }
    
    /** Calculate the size of the region surrounding the current object (eg. brain). */
    InputMaskImageIndexType minIndex;
    InputMaskImageIndexType maxIndex;
    InputMaskImageIndexType currentIndex;
    InputMaskImageRegionType imageRegion;    
    InputMaskImageSizeType  imageSize;
    InputMaskImageRegionType scanRegion;
    
    imageRegion = inputThresholdImage->GetLargestPossibleRegion();
    imageSize = imageRegion.GetSize();

    minIndex[0] = imageSize[0]-1;
    minIndex[1] = imageSize[1]-1;
    minIndex[2] = imageSize[2]-1;
    
    ImageRegionConstIteratorWithIndex<InputMaskImageType> forwardIterator(inputThresholdImage, imageRegion);   
    for (forwardIterator.GoToBegin(); !forwardIterator.IsAtEnd(); ++forwardIterator)
    {
      if (forwardIterator.Get() != m_OutValue)
      {
        currentIndex = forwardIterator.GetIndex();
        for (int i = 0; i < 3; i++)
        { 
          if (currentIndex[i] < minIndex[i]) minIndex[i] = currentIndex[i];
        }
      }
    }

    maxIndex.Fill(0);

    ImageRegionConstIteratorWithIndex<InputMaskImageType> backwardIterator(inputThresholdImage, imageRegion);   
    for (backwardIterator.GoToReverseBegin(); !backwardIterator.IsAtReverseEnd(); --backwardIterator)
    {
      if (backwardIterator.Get() != m_OutValue)
      {
        currentIndex = backwardIterator.GetIndex();
        for (int i = 0; i < 3; i++)
        { 
          if (currentIndex[i] > maxIndex[i]) maxIndex[i] = currentIndex[i];
        }
      }
    }
    
    InputMaskImageSizeType regionOfInterestSize;
    regionOfInterestSize[0] = maxIndex[0] - minIndex[0] + 1;
    regionOfInterestSize[1] = maxIndex[1] - minIndex[1] + 1;
    regionOfInterestSize[2] = maxIndex[2] - minIndex[2] + 1;
    
    InputMaskImageRegionType regionOfInterest;
    regionOfInterest.SetSize(regionOfInterestSize);
    regionOfInterest.SetIndex(minIndex);
    
    // The region of interest is the output of the first tab - the size of the thresholded brain
    // before we do any erosions and dilations and edits. See MorphologicalSegmentorPipeline::SetParam
     
    m_ROIImageFilter->SetInput(inputMaskImage);
    m_ROIImageFilter->SetRegionOfInterest(regionOfInterest);
    m_ROIImageFilter->UpdateLargestPossibleRegion();

    m_DownSamplingFilter->SetInput(m_ROIImageFilter->GetOutput());
    m_DownSamplingFilter->SetDownSamplingFactor(m_DownSamplingFactor);
    m_DownSamplingFilter->SetInValue(m_InValue);
    m_DownSamplingFilter->SetOutValue(m_OutValue);
    m_DownSamplingFilter->UpdateLargestPossibleRegion();

    StructuringElementType element;
    element.SetRadius(1);
    element.CreateStructuringElement();
    
    m_ErosionFilter1->SetInput(m_DownSamplingFilter->GetOutput());
    m_ErosionFilter1->SetKernel(element);
    m_ErosionFilter1->SetErodeValue(m_InValue);
    m_ErosionFilter1->SetBackgroundValue(m_OutValue);
    m_ErosionFilter1->UpdateLargestPossibleRegion();

    m_ErosionFilter2->SetInput(m_ErosionFilter1->GetOutput());
    m_ErosionFilter2->SetKernel(element);
    m_ErosionFilter2->SetErodeValue(m_InValue);
    m_ErosionFilter2->SetBackgroundValue(m_OutValue);
    m_ErosionFilter2->UpdateLargestPossibleRegion();

    m_UpSamplingFilter->SetInput(0, m_ErosionFilter2->GetOutput()); // this is the one we upsample.
    m_UpSamplingFilter->SetInput(1, m_ROIImageFilter->GetOutput()); // this is the one used to set the size, i.e. before we downsampled.
    m_UpSamplingFilter->SetInValue(m_InValue);
    m_UpSamplingFilter->SetOutValue(m_OutValue);
    m_UpSamplingFilter->SetUpSamplingFactor(m_DownSamplingFactor);
    m_UpSamplingFilter->UpdateLargestPossibleRegion();

    m_PasteImageFilter->SetInput(inputMaskImage);
    m_PasteImageFilter->SetSourceImage(m_UpSamplingFilter->GetOutput());
    m_PasteImageFilter->SetSourceRegion(m_UpSamplingFilter->GetOutput()->GetLargestPossibleRegion());
    m_PasteImageFilter->SetDestinationIndex(minIndex);
    m_PasteImageFilter->UpdateLargestPossibleRegion();

    // Calculate mean, and percentages using original grey scale image, 
    // and current segmentation (output of dilations tab).
    // Not using itkMIDASMeanIntensityWithinARegionFilter to save memory.
    
    float mean = 0;                     // These are deliberately float, because MIDAS is float.
    float actualLow = 0;
    float actualHigh = 0;
    unsigned long int counter = 0;
        
    ImageRegionConstIterator<InputMainImageType> inputImageIterator(inputMainImage, inputMainImage->GetLargestPossibleRegion());
    ImageRegionConstIterator<InputMaskImageType> inputImageMaskIterator(inputMaskImage, inputMaskImage->GetLargestPossibleRegion());
    for (inputImageIterator.GoToBegin(), inputImageMaskIterator.GoToBegin();
         !inputImageIterator.IsAtEnd();
         ++inputImageIterator, ++inputImageMaskIterator)
      {
        if (inputImageMaskIterator.Get() != m_OutValue)
        {
          counter++;
          mean += inputImageIterator.Get();
        }
      }
    if (counter != 0)
    {
      mean /= (double)counter;
    }    
    actualLow = mean * (m_LowPercentageThreshold/100.0);
    actualHigh = mean * (m_HighPercentageThreshold/100.0);

    OutputImagePointer outputImagePtr = this->GetOutput();
    
    ImageRegionConstIterator<OutputImageType> pasteImageIterator(m_PasteImageFilter->GetOutput(), m_PasteImageFilter->GetOutput()->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType> outputImageIterator(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());
    
    for (inputImageIterator.GoToBegin(),
         inputImageMaskIterator.GoToBegin(),
         pasteImageIterator.GoToBegin(),         
         outputImageIterator.GoToBegin();
         !inputImageIterator.IsAtEnd();
         ++inputImageIterator,
         ++inputImageMaskIterator,
         ++pasteImageIterator,
         ++outputImageIterator
         )
      {
        if (pasteImageIterator.Get() != m_OutValue)
        {
          if (inputImageIterator.Get() > actualLow && inputImageIterator.Get() < actualHigh)
          {
            outputImageIterator.Set(m_InValue);
          }
          else
          {
            outputImageIterator.Set(m_OutValue);
          }
        }
        else
        {
          outputImageIterator.Set(inputImageMaskIterator.Get());
        }
      }
  }
  
}//end namespace itk

#endif
