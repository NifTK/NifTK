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
    m_StructuringElementRadius = 2;
    m_LowPercentageThreshold = 50;
    m_HighPercentageThreshold = 150;
    m_DoFormulaThree = false;
    m_ROIImageFilter = ROIImageFilterType::New();
    m_DownSamplingFilter = DownSamplingFilterType::New();
    m_ErosionFilter1 = ErosionFilterType::New();
    m_ErosionFilter2 = ErosionFilterType::New();
    m_UpSamplingFilter = UpSamplingFilterType::New();
    m_PasteImageFilter = PasteImageFilterType::New();
    m_MeanFilter = MeanFilterType::New();
    m_ThresholdFilter = ThresholdFilterType::New();
    m_AndFilter = AndFilterType::New();
    m_OrFilter = OrFilterType::New();    
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASRethresholdingFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_DownSamplingFactor=" << m_DownSamplingFactor << std::endl;
    os << indent << "m_InValue=" << m_InValue << std::endl;
    os << indent << "m_OutValue=" << m_OutValue << std::endl;
    os << indent << "m_StructuringElementRadius=" << m_StructuringElementRadius << std::endl;
    os << indent << "m_LowPercentageThreshold=" << m_LowPercentageThreshold << std::endl;
    os << indent << "m_HighPercentageThreshold=" << m_HighPercentageThreshold << std::endl;
    os << indent << "m_DoFormulaThree=" << m_DoFormulaThree << std::endl;
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
     
    if(numberOfInputImages != 2)
    {
      niftkitkDebugMacro(<< "There should be two input images for MIDASRethresholdingFilter. ");
    }
    
    // Check input image is set.
    InputMainImageType *inputMainImage = static_cast<InputMainImageType*>(this->ProcessObject::GetInput(0));
    if(!inputMainImage)
    {
      niftkitkDebugMacro(<< "Input image is not set!");
    }
    
    // Check input binary mask is set.
    InputMaskImageType *inputMaskImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    if(!inputMaskImage)
    {
      niftkitkDebugMacro(<< "Input mask is not set!");
    }

    // Check input thresholded mask is set.
    InputMaskImageType *inputThresholdImage = static_cast<InputMaskImageType*>(this->ProcessObject::GetInput(2));
    if(!inputThresholdImage)
    {
      niftkitkDebugMacro(<< "Input threshold image is not set!");
    }

    // Check the sizes match.
    if( (inputMainImage->GetLargestPossibleRegion().GetSize()) != (inputMaskImage->GetLargestPossibleRegion().GetSize()) )
    { 
      niftkitkDebugMacro(<< "Input images don't match in size!");
    }
  
    this->AllocateOutputs();
    
    if (m_DownSamplingFactor <= 1)
    {
      this->CopyImageToOutput(inputMaskImage);
      return;
    }
    
    StructuringElementType element;
    element.SetRadius(m_StructuringElementRadius);
    element.CreateStructuringElement();

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
    
    m_ROIImageFilter->SetInput(inputMaskImage);
    m_ROIImageFilter->SetRegionOfInterest(regionOfInterest);
    m_ROIImageFilter->Update();
    
    m_DownSamplingFilter->SetInput(m_ROIImageFilter->GetOutput());
    m_DownSamplingFilter->SetDownSamplingFactor(m_DownSamplingFactor);
    m_DownSamplingFilter->SetInValue(m_InValue);
    m_DownSamplingFilter->SetOutValue(m_OutValue);
    m_DownSamplingFilter->Update();
    
    m_ErosionFilter1->SetInput(m_DownSamplingFilter->GetOutput());
    m_ErosionFilter1->SetKernel(element);
    m_ErosionFilter1->SetErodeValue(m_InValue);
    m_ErosionFilter1->SetBackgroundValue(m_OutValue);
    m_ErosionFilter1->SetBoundaryToForeground(false);
    m_ErosionFilter1->Update();

    m_ErosionFilter2->SetInput(m_ErosionFilter1->GetOutput());
    m_ErosionFilter2->SetKernel(element);
    m_ErosionFilter2->SetErodeValue(m_InValue);
    m_ErosionFilter2->SetBackgroundValue(m_OutValue);
    m_ErosionFilter2->SetBoundaryToForeground(false);
    m_ErosionFilter2->Update();
    
    m_UpSamplingFilter->SetInput(0, m_ErosionFilter2->GetOutput()); // this is the one we upsample.
    m_UpSamplingFilter->SetInput(1, m_ROIImageFilter->GetOutput()); // this is the one used to set the size.
    m_UpSamplingFilter->SetInValue(m_InValue);
    m_UpSamplingFilter->SetOutValue(m_OutValue);
    m_UpSamplingFilter->SetUpSamplingFactor(m_DownSamplingFactor);
    m_UpSamplingFilter->Update();

    m_PasteImageFilter->SetInput(inputMaskImage);
    m_PasteImageFilter->SetSourceImage(m_UpSamplingFilter->GetOutput());
    m_PasteImageFilter->SetSourceRegion(m_UpSamplingFilter->GetOutput()->GetLargestPossibleRegion());
    m_PasteImageFilter->SetDestinationIndex(minIndex);
    m_PasteImageFilter->Update();

    if (m_DoFormulaThree)
    {
      m_OrFilter->SetInput(0, inputMaskImage);
      m_OrFilter->SetInput(1, m_PasteImageFilter->GetOutput());
      m_OrFilter->Update();

      m_MeanFilter->SetGreyScaleImageInput(inputMainImage);
      m_MeanFilter->SetBinaryImageInput(inputMaskImage);
      m_MeanFilter->SetInValue(m_InValue);
      m_MeanFilter->Update();
      
      double mean = m_MeanFilter->GetMeanIntensityMainImage();
      double actualLow = mean * (m_LowPercentageThreshold/100.0);
      double actualHigh = mean * (m_HighPercentageThreshold/100.0);
      
      std::cerr << "GenerateData(): mean=" << mean << ", percentages=[" << m_LowPercentageThreshold << ", " << m_HighPercentageThreshold << "], actual=[" << actualLow << ", " << actualHigh << "]" <<std::endl;
  
      m_ThresholdFilter->SetInput(inputMainImage);
      m_ThresholdFilter->SetInsideValue(m_InValue);
      m_ThresholdFilter->SetOutsideValue(m_OutValue);
      m_ThresholdFilter->SetLowerThreshold((typename TInputImage1::PixelType)(actualLow));
      m_ThresholdFilter->SetUpperThreshold((typename TInputImage1::PixelType)(actualHigh));
      m_ThresholdFilter->Update();    
      
      m_AndFilter->SetInput(0, m_OrFilter->GetOutput());
      m_AndFilter->SetInput(1, m_ThresholdFilter->GetOutput());
      m_AndFilter->Update();
    }
    
    DONT USE PASTE FILTER, USE INJECT FILTER.
    
    if (m_DoFormulaThree)
    {
      this->CopyImageToOutput(m_AndFilter->GetOutput());
    }
    else
    {
      this->CopyImageToOutput(m_PasteImageFilter->GetOutput());
    }
    
    if (1)
    {
      typename itk::ImageFileWriter<InputMaskImageType>::Pointer maskWriter = itk::ImageFileWriter<InputMaskImageType>::New();
      maskWriter->SetInput(inputMaskImage);
      maskWriter->SetFileName("tmp.original.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_ROIImageFilter->GetOutput());
      maskWriter->SetFileName("tmp.roi.nii");
      maskWriter->Update();
      
      maskWriter->SetInput(m_DownSamplingFilter->GetOutput());
      maskWriter->SetFileName("tmp.down.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_ErosionFilter1->GetOutput());
      maskWriter->SetFileName("tmp.eroded.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_ErosionFilter2->GetOutput());
      maskWriter->SetFileName("tmp.eroded2.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_UpSamplingFilter->GetOutput());
      maskWriter->SetFileName("tmp.up.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_PasteImageFilter->GetOutput());
      maskWriter->SetFileName("tmp.paste.nii");
      maskWriter->Update();
    
      if (m_DoFormulaThree)
      {
        maskWriter->SetInput(m_OrFilter->GetOutput());
        maskWriter->SetFileName("tmp.or.nii");
        maskWriter->Update();

        maskWriter->SetInput(m_ThresholdFilter->GetOutput());
        maskWriter->SetFileName("tmp.thresh.nii");
        maskWriter->Update();
  
        maskWriter->SetInput(m_AndFilter->GetOutput());
        maskWriter->SetFileName("tmp.and.nii");
        maskWriter->Update();      
      }      
    } 
  }
  
}//end namespace itk

#endif
