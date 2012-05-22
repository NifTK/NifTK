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
  m_DilationFilter = DilationFilterType::New();
  m_RethresholdingFilter = RethresholdingFilterType::New();
  m_LateMaskFilter = MaskByRegionFilterType::New();
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
  m_DilationFilter->SetInValue(m_ForegroundValue);
  m_DilationFilter->SetOutValue(m_BackgroundValue);
  m_RethresholdingFilter->SetInValue(m_ForegroundValue);
  m_RethresholdingFilter->SetOutValue(m_BackgroundValue);
  m_LateMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
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
    m_LateMaskFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_LateMaskFilter->GetOutput());

    m_ErosionFilter->SetUpperThreshold((TPixel)p.m_UpperErosionsThreshold);
    m_ErosionFilter->SetNumberOfIterations(p.m_NumberOfErosions);
  }
  else if (m_Stage == 2)
  {
    m_EarlyMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
    m_EarlyConnectedComponentFilter->SetInput(m_EarlyMaskFilter->GetOutput());
    m_ErosionFilter->SetBinaryImageInput(m_EarlyConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationFilter->SetBinaryImageInput(m_ErosionFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_LateMaskFilter->SetInput(0, m_DilationFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_LateMaskFilter->GetOutput());

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
    m_DilationFilter->SetBinaryImageInput(m_ErosionFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_RethresholdingFilter->SetBinaryImageInput(m_DilationFilter->GetOutput());
    m_RethresholdingFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_LateMaskFilter->SetInput(0, m_RethresholdingFilter->GetOutput());
    m_LateConnectedComponentFilter->SetInput(m_LateMaskFilter->GetOutput());

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
    if (additionsImageBeingEdited)
    {
      // Note: This little... Hacklet.. or shall we say "optimisation", basically replicates
      // the filter logic, over a tiny region of interest. I did try using filters to extract
      // a region of interest, perform the logic in another filter, and then insert the region
      // back, but it didn't work, even after sacrificing virgins to several well known deities.

      itk::ImageRegionIterator<SegmentationImageType> outputIterator(m_LateMaskFilter->GetOutput(), editingRegionOfInterest);
      itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(m_LateMaskFilter->GetInput(1), editingRegionOfInterest);
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
      // Note: This little... Hacklet.. or shall we say "optimisation", basically replicates
      // the filter logic, over a tiny region of interest. I did try using filters to extract
      // a region of interest, perform the logic in another filter, and then insert the region
      // back, but it didn't work, even after sacrificing virgins to several well known deities.

      itk::ImageRegionIterator<SegmentationImageType> outputIterator(m_LateMaskFilter->GetOutput(), editingRegionOfInterest);
      itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(m_LateMaskFilter->GetInput(2), editingRegionOfInterest);
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
    else
    {
      // Executing the pipeline for the whole image - slow, but unavoidable.
      m_LateConnectedComponentFilter->Modified();
      m_LateConnectedComponentFilter->UpdateLargestPossibleRegion();
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
typename MorphologicalSegmentorPipeline<TPixel, VImageDimension>::SegmentationImageType::Pointer
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::GetOutput(bool editingImageBeingEdited, bool additionsImageBeingEdited)
{
  typename SegmentationImageType::Pointer result;

  if (m_Stage == 0)
  {
    result = m_EarlyMaskFilter->GetOutput();
  }
  else
  {
    if (additionsImageBeingEdited || editingImageBeingEdited)
    {
      result = m_LateMaskFilter->GetOutput();
    }
    else
    {
      result = m_LateConnectedComponentFilter->GetOutput();
    }
  }
  return result;
}

