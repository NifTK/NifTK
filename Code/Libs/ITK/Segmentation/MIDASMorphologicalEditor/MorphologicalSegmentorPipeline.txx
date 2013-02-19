/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MorphologicalSegmentorPipeline.h"
#include "itkConversionUtils.h"
#include "itkMIDASHelper.h"

template<typename TPixel, unsigned int VImageDimension>
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::MorphologicalSegmentorPipeline()
{
  unsigned long int capacity = 2000000;
  
  // This is the main pipeline that will form the whole of the final output.
  m_ThresholdingFilter = ThresholdingFilterType::New();
  m_ThresholdingMaskFilter = MaskByRegionFilterType::New();
  m_ThresholdingConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_ThresholdingConnectedComponentFilter->SetCapacity(capacity);
  m_ErosionFilter = ErosionFilterType::New();
  m_ErosionMaskFilter = MaskByRegionFilterType::New();
  m_ErosionConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_ErosionConnectedComponentFilter->SetCapacity(capacity);
  m_DilationFilter = DilationFilterType::New();
  m_DilationMaskFilter = MaskByRegionFilterType::New();
  m_DilationConnectedComponentFilter = LargestConnectedComponentFilterType::New();
  m_DilationConnectedComponentFilter->SetCapacity(capacity);
  m_RethresholdingFilter = RethresholdingFilterType::New();

  this->SetForegroundValue((unsigned char)1);
  this->SetBackgroundValue((unsigned char)0);
}

template<typename TPixel, unsigned int VImageDimension>
void 
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::SetForegroundValue(unsigned char foregroundValue)
{
  m_ForegroundValue = foregroundValue;
  this->UpdateForegroundValues();
}


template<typename TPixel, unsigned int VImageDimension>
void 
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::SetBackgroundValue(unsigned char backgroundValue)
{
  m_BackgroundValue = backgroundValue;
  this->UpdateBackgroundValues();
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension> 
::UpdateForegroundValues()
{
  m_ThresholdingFilter->SetInsideValue(m_ForegroundValue);
  m_ThresholdingConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
  m_ErosionFilter->SetInValue(m_ForegroundValue);
  m_ErosionConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);  
  m_DilationFilter->SetInValue(m_ForegroundValue);
  m_DilationConnectedComponentFilter->SetOutputForegroundValue(m_ForegroundValue);
  m_RethresholdingFilter->SetInValue(m_ForegroundValue);
}

template<typename TPixel, unsigned int VImageDimension>  
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>  
::UpdateBackgroundValues()
{
  m_ThresholdingFilter->SetOutsideValue(m_BackgroundValue);
  m_ThresholdingMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_ThresholdingConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_ThresholdingConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_ErosionFilter->SetOutValue(m_BackgroundValue);
  m_ErosionMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_ErosionConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_ErosionConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_DilationFilter->SetOutValue(m_BackgroundValue);
  m_DilationMaskFilter->SetOutputBackgroundValue(m_BackgroundValue);
  m_DilationConnectedComponentFilter->SetInputBackgroundValue(m_BackgroundValue);
  m_DilationConnectedComponentFilter->SetOutputBackgroundValue(m_BackgroundValue);;
  m_RethresholdingFilter->SetOutValue(m_BackgroundValue);
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::DisconnectPipeline()
{
  // Aim: Make sure all smart pointers to the input reference (grey scale T1 image) are released.
  m_ThresholdingFilter->SetInput(NULL);
  m_ErosionFilter->SetGreyScaleImageInput(NULL);
  m_DilationFilter->SetGreyScaleImageInput(NULL);
  m_RethresholdingFilter->SetGreyScaleImageInput(NULL);

  m_ErosionMaskFilter->SetInput(1, NULL);
  m_ErosionMaskFilter->SetInput(2, NULL);
  m_DilationMaskFilter->SetInput(1, NULL);
  m_DilationMaskFilter->SetInput(2, NULL);
  m_DilationFilter->SetConnectionBreakerImage(NULL);
}


template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::SetParam(GreyScaleImageType* referenceImage,
    SegmentationImageType* erosionsAdditionsImage,
    SegmentationImageType* erosionEditsImage,
    SegmentationImageType* dilationsAditionsImage,
    SegmentationImageType* dilationsEditsImage,
    MorphologicalSegmentorPipelineParams& p)
{
  m_Stage = p.m_Stage;

  // Connect input images.
  m_ThresholdingFilter->SetInput(referenceImage);
  m_ErosionFilter->SetGreyScaleImageInput(referenceImage);
  m_DilationFilter->SetGreyScaleImageInput(referenceImage);
  m_RethresholdingFilter->SetGreyScaleImageInput(referenceImage);

  m_ErosionMaskFilter->SetInput(1, erosionsAdditionsImage);
  m_ErosionMaskFilter->SetInput(2, erosionEditsImage);
  m_DilationMaskFilter->SetInput(1, dilationsAditionsImage);
  m_DilationMaskFilter->SetInput(2, dilationsEditsImage);
  m_DilationFilter->SetConnectionBreakerImage(dilationsEditsImage);

  // Note, the ITK Set/Get Macro ensures that the Modified flag only gets set if the value set is actually different.

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Start Trac 998, setting region of interest, on all Mask filters, to produce Axial-Cut-off effect.
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  typename SegmentationImageType::RegionType         regionOfInterest;
  typename SegmentationImageType::SizeType           regionOfInterestSize;
  typename SegmentationImageType::IndexType          regionOfInterestIndex;
  typedef typename GreyScaleImageType::ConstPointer  ImagePointer;
  
  ImagePointer input = m_ThresholdingFilter->GetInput();
  
  // 1. Set region to full size of input image
  regionOfInterestSize = input->GetLargestPossibleRegion().GetSize();
  regionOfInterestIndex = input->GetLargestPossibleRegion().GetIndex();

  // 2. Get string describing orientation.
  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation;
  orientation = adaptor.FromDirectionCosines(input->GetDirection());
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientation);

  // 3. Get Axis that represents superior/inferior
  int axialAxis = -1;
  itk::ORIENTATION_ENUM orientationEnum = itk::ORIENTATION_AXIAL; 
  itk::GetAxisFromITKImage<TPixel, VImageDimension>(input, orientationEnum, axialAxis);
  
  if (axialAxis != -1)
  {
    // 4. Calculate size of region of interest in that axis
    regionOfInterestSize[axialAxis] = regionOfInterestSize[axialAxis] - p.m_AxialCutoffSlice;
    if (orientationString[axialAxis] == 'I')
    {
      regionOfInterestIndex[axialAxis] = p.m_AxialCutoffSlice;
    }
    else
    {
      regionOfInterestIndex[axialAxis] = 0;
    }

    // 5. Set region on both filters
    regionOfInterest.SetSize(regionOfInterestSize);
    regionOfInterest.SetIndex(regionOfInterestIndex);
    
    if (regionOfInterest != this->m_ThresholdingMaskFilter->GetRegion())
    {
      this->m_ThresholdingMaskFilter->SetRegion(regionOfInterest);
    }

    if (regionOfInterest != this->m_ErosionMaskFilter->GetRegion())
    { 
      this->m_ErosionFilter->SetRegion(regionOfInterest);
      this->m_ErosionMaskFilter->SetRegion(regionOfInterest);
    }
    
    if (regionOfInterest != this->m_DilationMaskFilter->GetRegion())
    {        
      this->m_DilationMaskFilter->SetRegion(regionOfInterest);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // End Trac 998, setting region of interest, on Mask filters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Start Trac 1131, calculate a rough size to help LargestConnectedComponents allocate memory.
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  unsigned long int expectedSize = 1;
  for (unsigned int i = 0; i < VImageDimension; i++)
  {
    expectedSize *= regionOfInterestSize[i];
  }
  expectedSize /= 8;

  m_ThresholdingConnectedComponentFilter->SetCapacity(expectedSize);
  m_ErosionConnectedComponentFilter->SetCapacity(expectedSize);
  m_DilationConnectedComponentFilter->SetCapacity(expectedSize);
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // End Trac 1131.
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  if (m_Stage == 0)
  {
    m_ThresholdingFilter->SetLowerThreshold((TPixel)p.m_LowerIntensityThreshold);
    m_ThresholdingFilter->SetUpperThreshold((TPixel)p.m_UpperIntensityThreshold);

    m_ThresholdingMaskFilter->SetInput(m_ThresholdingFilter->GetOutput());
  }
  else if (m_Stage == 1)
  {

    m_ThresholdingConnectedComponentFilter->SetInput(m_ThresholdingMaskFilter->GetOutput());

    m_ErosionFilter->SetBinaryImageInput(m_ThresholdingConnectedComponentFilter->GetOutput());
    m_ErosionFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_ErosionFilter->SetUpperThreshold((TPixel)p.m_UpperErosionsThreshold);
    m_ErosionFilter->SetNumberOfIterations(p.m_NumberOfErosions);

    m_ErosionMaskFilter->SetInput(0, m_ErosionFilter->GetOutput());
    m_ErosionConnectedComponentFilter->SetInput(m_ErosionMaskFilter->GetOutput());
  }
  else if (m_Stage == 2)
  {

    m_DilationFilter->SetBinaryImageInput(m_ErosionConnectedComponentFilter->GetOutput());
    m_DilationFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_DilationFilter->SetLowerThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_DilationFilter->SetUpperThreshold((int)(p.m_UpperPercentageThresholdForDilations));
    m_DilationFilter->SetNumberOfIterations((int)(p.m_NumberOfDilations));

    m_DilationMaskFilter->SetInput(0, m_DilationFilter->GetOutput());
    m_DilationConnectedComponentFilter->SetInput(m_DilationMaskFilter->GetOutput());
  }
  else if (m_Stage == 3)
  {
    m_RethresholdingFilter->SetBinaryImageInput(m_DilationConnectedComponentFilter->GetOutput());
    m_RethresholdingFilter->SetGreyScaleImageInput(m_ThresholdingFilter->GetInput());
    m_RethresholdingFilter->SetThresholdedImageInput(m_ThresholdingMaskFilter->GetOutput());
    m_RethresholdingFilter->SetDownSamplingFactor(p.m_BoxSize);
    m_RethresholdingFilter->SetLowPercentageThreshold((int)(p.m_LowerPercentageThresholdForDilations));
    m_RethresholdingFilter->SetHighPercentageThreshold((int)(p.m_UpperPercentageThresholdForDilations));
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::Update(std::vector<bool>& editingFlags, std::vector<int>& editingRegion)
{
  if (m_Stage == 0)
  {
    m_ThresholdingMaskFilter->Modified();
    m_ThresholdingMaskFilter->UpdateLargestPossibleRegion();
  }
  else if (m_Stage == 1 || m_Stage == 2)
  {
    // Simple case first - no editing.
    
    if (!editingFlags[0] && !editingFlags[1] && !editingFlags[2] && !editingFlags[3])
    {
      if (m_Stage == 1)
      {
        m_ErosionMaskFilter->GetInput()->Modified();
        m_ErosionMaskFilter->Modified();
        m_ErosionConnectedComponentFilter->Modified();
        m_ErosionConnectedComponentFilter->UpdateLargestPossibleRegion();
      }
      else if (m_Stage == 2)
      {
        m_DilationMaskFilter->GetInput()->Modified();
        m_DilationMaskFilter->Modified();
        m_DilationConnectedComponentFilter->Modified();
        m_DilationConnectedComponentFilter->UpdateLargestPossibleRegion();
      }
    }
    else
    {
      // Else: We are doing live updates.
      // Note: We try and update as small a section of the pipeline as possible - as GUI has to be interactive.

      typename SegmentationImageType::Pointer outputImage = NULL;
      typename SegmentationImageType::ConstPointer inputImage = NULL;
      int updateMethod = 0;
      
      typedef itk::Image<TPixel, VImageDimension> ImageType;
      typedef typename ImageType::IndexType       IndexType;
      typedef typename ImageType::SizeType        SizeType;
      typedef typename ImageType::RegionType      RegionType;
    
      IndexType  editingRegionStartIndex;
      SizeType   editingRegionSize;
      RegionType editingRegionOfInterest;
    
      for (int i = 0; i < 3; i++)
      {
        editingRegionStartIndex[i] = editingRegion[i];
        editingRegionSize[i] = editingRegion[i + 3];
      }
      editingRegionOfInterest.SetIndex(editingRegionStartIndex);
      editingRegionOfInterest.SetSize(editingRegionSize);
  
      // If Either of these are set, we must be in erosions tab, so m_Stage == 1.
      if (editingFlags[0] || editingFlags[1])
      {
        if (editingFlags[0])
        {
          inputImage = m_ErosionMaskFilter->GetInput(1);
          outputImage = m_ErosionConnectedComponentFilter->GetOutput();
          updateMethod = 1;        
        }
        else if (editingFlags[1])
        {
          inputImage = m_ErosionMaskFilter->GetInput(2);
          outputImage = m_ErosionConnectedComponentFilter->GetOutput();
          updateMethod = 2;                
        }
      }
      else // We are on dilations tab, so m_Stage == 2.
      {
        if (editingFlags[2])
        {
          inputImage = m_DilationMaskFilter->GetInput(1);
          outputImage = m_DilationConnectedComponentFilter->GetOutput();
          updateMethod = 1;                
        }
        else if (editingFlags[3])
        {
          inputImage = m_DilationMaskFilter->GetInput(2);
          outputImage = m_DilationConnectedComponentFilter->GetOutput();  
          updateMethod = 2;                      
        }      
      }
      // Now we have decided, input, output, and which method.
      if (updateMethod == 1)
      {
        itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(inputImage, editingRegionOfInterest);
        itk::ImageRegionIterator<SegmentationImageType> outputIterator(outputImage, editingRegionOfInterest);
        
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
      else if (updateMethod == 2)
      {
        itk::ImageRegionConstIterator<SegmentationImageType> editedRegionIterator(inputImage, editingRegionOfInterest);
        itk::ImageRegionIterator<SegmentationImageType> outputIterator(outputImage, editingRegionOfInterest);

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
  }
  else if (m_Stage == 3)
  {  
    m_RethresholdingFilter->Modified();
    m_RethresholdingFilter->UpdateLargestPossibleRegion();    
  }
}

template<typename TPixel, unsigned int VImageDimension>
typename MorphologicalSegmentorPipeline<TPixel, VImageDimension>::SegmentationImageType::Pointer
MorphologicalSegmentorPipeline<TPixel, VImageDimension>
::GetOutput(std::vector<bool>& editingFlags)
{
  typename SegmentationImageType::Pointer result;

  if (m_Stage == 0)
  {
    result = m_ThresholdingMaskFilter->GetOutput();
  }
  else if (m_Stage == 1)
  {
    result = m_ErosionConnectedComponentFilter->GetOutput();
  }
  else if (m_Stage == 2)
  {
    result = m_DilationConnectedComponentFilter->GetOutput();
  }
  else if (m_Stage == 3)
  {
    result = m_RethresholdingFilter->GetOutput();
  }
  return result;
}

