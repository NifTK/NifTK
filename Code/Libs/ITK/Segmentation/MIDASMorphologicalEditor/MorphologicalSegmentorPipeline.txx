/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "MorphologicalSegmentorPipeline.h"


template<typename TPixel, unsigned int VImageDimension>
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::MorphologicalSegmentorPipeline()
{
  unsigned long int capacity = 2000000;
  
  // This is the main pipeline that will form the whole of the final output.
  m_ThresholdingFilter = ThresholdingFilterType::New();
  m_EarlyMaskFilter = MaskByRegionFilterType::New();
  m_EarlyConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_EarlyConnectedComponentFilter->SetCapacity(capacity);
  m_ErosionFilter = ErosionFilterType::New();
  m_ErosionMaskFilter = MaskByRegionFilterType::New();
  m_DilationFilter = DilationFilterType::New();
  m_DilationMaskFilter = MaskByRegionFilterType::New();
  m_RethresholdingFilter = RethresholdingFilterType::New();
  m_LateConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_LateConnectedComponentFilter->SetCapacity(capacity);

  // Making sure that these are only called once in constructor, to avoid unnecessary pipeline updates.
  m_ForegroundValue = 255;
  m_BackgroundValue = 0;
  m_ThresholdingFilter->SetInsideValue(m_ForegroundValue);
  m_ThresholdingFilter->SetOutsideValue(m_BackgroundValue);
  m_EarlyMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_EarlyConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
  m_ErosionFilter->SetInValue(m_ForegroundValue);
  m_ErosionFilter->SetOutValue(m_BackgroundValue);
  m_ErosionMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_DilationFilter->SetInValue(m_ForegroundValue);
  m_DilationFilter->SetOutValue(m_BackgroundValue);
  m_DilationMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_RethresholdingFilter->SetInValue(m_ForegroundValue);
  m_RethresholdingFilter->SetOutValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_LateConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::SetParam(MorphologicalSegmentorPipelineParams& p)
{
  m_Stage = p.m_Stage;

  // Note, the ITK Set/Get Macro ensures that the Modified flag only gets set if the value set is actually different.

  if (m_Stage == 0)
  {
    m_ThresholdingFilter->SetLowerThreshold((TPixel)p.m_LowerIntensityThreshold);
    m_ThresholdingFilter->SetUpperThreshold((TPixel)p.m_UpperIntensityThreshold);

    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
  }
  else if (m_Stage == 1)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_ErosionMaskFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ErosionMaskFilter->GetOutput());

    m_ErosionFilter->SetUpperThreshold((TPixel)p.m_UpperErosionsThreshold);
    m_ErosionFilter->SetNumberOfIterations(p.m_NumberOfErosions);
  }
  else if (m_Stage == 2)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_ErosionMaskFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ErosionMaskFilter->GetOutput());
    m_DilationFilter->SetBinaryImageInput(m_LateConnectedComponentFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationMaskFilter->SetInput(0, m_DilationFilter->GetOutput());
    
    m_DilationFilter->SetLowerThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_DilationFilter->SetUpperThreshold((int)(p.m_UpperPercentageThresholdForDilations));
    m_DilationFilter->SetNumberOfIterations((int)(p.m_NumberOfDilations));
  }
  else if (m_Stage == 3)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_ErosionMaskFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_ErosionMaskFilter->GetOutput());
    m_DilationFilter->SetBinaryImageInput(m_LateConnectedComponentFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationMaskFilter->SetInput(0, m_DilationFilter->GetOutput());
    m_RethresholdingFilter->SetBinaryImageInput(m_DilationMaskFilter->GetOutput());
    m_RethresholdingFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    
    m_RethresholdingFilter->SetDownSamplingFactor(p.m_BoxSize);
    m_RethresholdingFilter->SetLowPercentageThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_RethresholdingFilter->SetHighPercentageThreshold((int)(p.m_UpperPercentageThresholdForDilations));
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::Update(bool editingImageBeingEdited, bool additionsImageBeingEdited, std::vector<int>& editingRegion)
{
  // Note: We try and update as small a section of the pipeline as possible.

  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::RegionType RegionType;

  IndexType editingRegionStartIndex;
  SizeType editingRegionSize;
  RegionType editingRegionOfInterest;

  for (int i = 0; i < 3; i++)
  {
    editingRegionStartIndex[i] = editingRegion[i];
    editingRegionSize[i] = editingRegion[i + 3];
  }
  editingRegionOfInterest.SetIndex(editingRegionStartIndex);
  editingRegionOfInterest.SetSize(editingRegionSize);

  if (m_Stage == 0)
  {
    m_EarlyMaskFilter->UpdateLargestPossibleRegion();
  }
  else
  {

    if (additionsImageBeingEdited || editingImageBeingEdited)
    {
      typename SegmentationImageType::Pointer outputImage = NULL;
      typename SegmentationImageType::ConstPointer inputImage = NULL;
      
      int inputNumber = 0;
      if (additionsImageBeingEdited)
      {
        inputNumber = 1;
      }
      else if (editingImageBeingEdited)
      {
        inputNumber = 2;
      }
      
      if (m_Stage == 1)
      {
        inputImage = m_ErosionMaskFilter->GetInput(inputNumber);
        outputImage = m_LateConnectedComponentFilter->GetOutput();
      }
      else if (m_Stage == 2)
      {
        inputImage = m_DilationMaskFilter->GetInput(inputNumber);
        outputImage = m_DilationMaskFilter->GetOutput();
      }
      else if (m_Stage == 3)
      {
        outputImage = m_RethresholdingFilter->GetOutput();
      }
    
      if (additionsImageBeingEdited)
      {
        itk::ImageRegionIterator<SegmentationImageType> outputIterator(outputImage, editingRegionOfInterest);
        itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(inputImage, editingRegionOfInterest);
        for (outputIterator.GoToBegin(), editedRegionIterator.GoToBegin();
            !outputIterator.IsAtEnd();
            ++outputIterator, ++editedRegionIterator)
        {
          if (outputIterator.Get() > 0 || editedRegionIterator.Get() > 0)
          {
            outputIterator.Set(m_ForegroundValue);
          }
          else
          {
            outputIterator.Set(m_BackgroundValue);
          }
        }
      }
      else if (editingImageBeingEdited)
      {
        itk::ImageRegionIterator<SegmentationImageType> outputIterator(outputImage, editingRegionOfInterest);
        itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(inputImage, editingRegionOfInterest);
        for (outputIterator.GoToBegin(), editedRegionIterator.GoToBegin();
            !outputIterator.IsAtEnd();
            ++outputIterator, ++editedRegionIterator)
        {
          if (editedRegionIterator.Get() > 0)
          {
            outputIterator.Set(m_BackgroundValue);
          }
        }
      }
    }
    else if (m_Stage == 1)
    {
      m_LateConnectedComponentFilter->Modified();
      m_LateConnectedComponentFilter->UpdateLargestPossibleRegion();
    }
    else if (m_Stage == 2)
    {
      m_DilationFilter->Modified();
      m_DilationFilter->UpdateLargestPossibleRegion();
    }
    else if (m_Stage == 3)
    {
      m_RethresholdingFilter->Modified();
      m_RethresholdingFilter->UpdateLargestPossibleRegion();    
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
typename MorphologicalSegmentorPipeline<TPixel, VImageDimension>::SegmentationImageType::Pointer
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::GetOutput(bool editingImageBeingEdited, bool additionsImageBeingEdited)
{
  typename SegmentationImageType::Pointer result;

  if (additionsImageBeingEdited || editingImageBeingEdited)
  {
    if (m_Stage == 1)
    {
      result = m_LateConnectedComponentFilter->GetOutput();
    }
    else if (m_Stage == 2)
    {
      result = m_DilationMaskFilter->GetOutput();
    }
    else if (m_Stage == 3)
    {
      result = m_RethresholdingFilter->GetOutput();
    }
  }
  else if (m_Stage == 0)
  {
    result = m_EarlyMaskFilter->GetOutput();
  }
  else if (m_Stage == 1)
  {
    result = m_LateConnectedComponentFilter->GetOutput();
  }
  else if (m_Stage == 2)
  {
    result = m_DilationFilter->GetOutput();
  }
  else if (m_Stage == 3)
  {
    result = m_RethresholdingFilter->GetOutput();    
  }
  return result;
}

