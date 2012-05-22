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
    m_SkipIntersectionWithMeanMask = false;
    m_DownSamplingFilter = DownSamplingFilterType::New();
    m_ErosionFilter = ErosionFilterType::New();
    m_UpSamplingFilter = UpSamplingFilterType::New();
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
    os << indent << "m_SkipIntersectionWithMeanMask=" << m_SkipIntersectionWithMeanMask << std::endl;
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
        
    m_DownSamplingFilter->SetInput(inputMaskImage);
    m_DownSamplingFilter->SetDownSamplingFactor(m_DownSamplingFactor);
    m_DownSamplingFilter->SetInValue(m_InValue);
    m_DownSamplingFilter->SetOutValue(m_OutValue);
    
    m_ErosionFilter->SetInput(m_DownSamplingFilter->GetOutput());
    m_ErosionFilter->SetKernel(element);
    m_ErosionFilter->SetErodeValue(m_InValue);
    m_ErosionFilter->SetBackgroundValue(m_OutValue);
    m_ErosionFilter->SetBoundaryToForeground(false);
    m_ErosionFilter->Update();
    
    m_UpSamplingFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_UpSamplingFilter->SetInput(1, inputMaskImage);
    m_UpSamplingFilter->SetInValue(m_InValue);
    m_UpSamplingFilter->SetOutValue(m_OutValue);
    m_UpSamplingFilter->SetUpSamplingFactor(m_DownSamplingFactor);
    m_UpSamplingFilter->Update();

    m_OrFilter->SetInput(0, inputMaskImage);
    m_OrFilter->SetInput(1, m_UpSamplingFilter->GetOutput());
    m_OrFilter->Update();

    if (!m_SkipIntersectionWithMeanMask)
    {
      m_MeanFilter->SetGreyScaleImageInput(inputMainImage);
      m_MeanFilter->SetBinaryImageInput(inputMaskImage);
      m_MeanFilter->SetInValue(m_InValue);
      m_MeanFilter->Update();
      
      double mean = m_MeanFilter->GetMeanIntensityMainImage();
      double actualLow = mean * (m_LowPercentageThreshold/100.0);
      double actualHigh = mean * (m_HighPercentageThreshold/100.0);
      
      niftkitkDebugMacro(<< "GenerateData(): mean=" << mean << ", percentages=[" << m_LowPercentageThreshold << ", " << m_HighPercentageThreshold << "], actual=[" << actualLow << ", " << actualHigh << "]" );
  
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
    
    if (!m_SkipIntersectionWithMeanMask)
    {
      this->CopyImageToOutput(m_AndFilter->GetOutput());
    }
    else
    {
      this->CopyImageToOutput(m_OrFilter->GetOutput());
    }
    
    if (0)
    {
      typename itk::ImageFileWriter<InputMaskImageType>::Pointer maskWriter = itk::ImageFileWriter<InputMaskImageType>::New();
      maskWriter->SetInput(inputMaskImage);
      maskWriter->SetFileName("tmp.original.nii");
      maskWriter->Update();
      
      maskWriter->SetInput(m_DownSamplingFilter->GetOutput());
      maskWriter->SetFileName("tmp.down.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_ErosionFilter->GetOutput());
      maskWriter->SetFileName("tmp.eroded.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_UpSamplingFilter->GetOutput());
      maskWriter->SetFileName("tmp.up.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_ThresholdFilter->GetOutput());
      maskWriter->SetFileName("tmp.thresh.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_OrFilter->GetOutput());
      maskWriter->SetFileName("tmp.or.nii");
      maskWriter->Update();

      maskWriter->SetInput(m_AndFilter->GetOutput());
      maskWriter->SetFileName("tmp.and.nii");
      maskWriter->Update();
    } 
  }
  
}//end namespace itk

#endif
