/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBOUNDARYSHIFTINEGRALCALCULATOR_TXX
#define ITKBOUNDARYSHIFTINEGRALCALCULATOR_TXX

#include "itkBoundaryShiftIntegralCalculator.h"
#include "itkBinaryIntersectWithPaddingImageFilter.h"
#include "itkBinaryUnionWithPaddingImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkXorImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkBasicImageComparisonFunctions.h"
#include "itkMultipleDilateImageFilter.h"
#include "itkMultipleErodeImageFilter.h"
#include "algorithm"
#include "iostream"

namespace itk
{

template <class TInputImage, class TInputMask, class TOutputImage>
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::BoundaryShiftIntegralCalculator()
{
  m_BoundaryShiftIntegral = 0.0;
  m_NumberOfErosion = 1;
  m_NumberOfDilation = 1;
  m_BaselineIntensityNormalisationFactor = 1.0;
  m_RepeatIntensityNormalisationFactor = 1.0;
  m_UpperCutoffValue = 0.75;
  m_LowerCutoffValue = 0.25;
  m_PaddingValue = 0;
  m_NumberOfSubROIDilation = 1;
}

template <class TInputImage, class TInputMask, class TOutputImage>
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::~BoundaryShiftIntegralCalculator()
{
}

template <class TInputImage, class TInputMask, class TOutputImage>
void  
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::Compute()
{
  // Check for same image sizes and voxel sizes. 
  if (!IsSameRegionSize<TInputImage, TInputMask>(this->m_BaselineImage, this->m_BaselineMask) ||
      !IsSameRegionSize<TInputMask, TInputImage>(this->m_BaselineMask, this->m_RepeatImage) ||
      !IsSameRegionSize<TInputImage, TInputMask>(this->m_RepeatImage, this->m_RepeatMask) ||
      (this->m_SubROIMask.IsNotNull() && !IsSameRegionSize<TInputMask, TInputMask>(this->m_RepeatMask, this->m_SubROIMask)))
  {
    std::cerr <<"The images/masks do not have the same number of voxels in each dimension." << std::endl; 
  }
  if (!IsSameVoxelSize<TInputImage, TInputMask>(this->m_BaselineImage, this->m_BaselineMask) ||
      !IsSameVoxelSize<TInputMask, TInputImage>(this->m_BaselineMask, this->m_RepeatImage) ||
      !IsSameVoxelSize<TInputImage, TInputMask>(this->m_RepeatImage, this->m_RepeatMask) ||
      (this->m_SubROIMask.IsNotNull() && !IsSameVoxelSize<TInputMask, TInputMask>(this->m_RepeatMask, this->m_SubROIMask)))
  {
    std::cerr <<"The images/masks do not have the same voxel size." << std::endl; 
  }
	
  // Compute BSI mask.
  ComputeBSIMask();
  
  // Compute the BSI value. 
  IntegrateOverBSIMask();
}

template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "m_BoundaryShiftIntegral: " << m_BoundaryShiftIntegral << std::endl;
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeErodedIntersectMask(void)
{
  typedef BinaryIntersectWithPaddingImageFilter<TInputMask, TInputMask> IntersectFilterType;
  typename IntersectFilterType::Pointer intersectFilter = IntersectFilterType::New();
  
  // Intersect the two masks.
  intersectFilter->SetInput1(m_BaselineMask);
  intersectFilter->SetInput2(m_RepeatMask);
  intersectFilter->SetPaddingValue(m_PaddingValue);
  
  typedef MultipleErodeImageFilter<TInputMask> MultipleErodeImageFilterType;
  typename MultipleErodeImageFilterType::Pointer multipleErodeImageFilter = MultipleErodeImageFilterType::New();
  
  // Erode multiple times. 
  multipleErodeImageFilter->SetNumberOfErosions(this->m_NumberOfErosion);
  multipleErodeImageFilter->SetInput(intersectFilter->GetOutput());
  multipleErodeImageFilter->Update();
  m_ErodedIntersectMask = multipleErodeImageFilter->GetOutput();
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeDilatedUnionMask(void)
{
  typedef BinaryUnionWithPaddingImageFilter<TInputMask, TInputMask> UnionFilterType;
  typename UnionFilterType::Pointer unionFilter = UnionFilterType::New();
  
  // Union the two masks.
  unionFilter->SetInput1(m_BaselineMask);
  unionFilter->SetInput2(m_RepeatMask);
  unionFilter->SetPaddingValue(m_PaddingValue);
  
  typedef MultipleDilateImageFilter<TInputMask> MultipleDilateImageFilterType;
  typename MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  
  // Dilate multiple times. 
  multipleDilateImageFilter->SetNumberOfDilations(this->m_NumberOfDilation);
  multipleDilateImageFilter->SetInput(unionFilter->GetOutput());
  multipleDilateImageFilter->Update();
  m_DilatedUnionMask = multipleDilateImageFilter->GetOutput();
}

template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeBSIMask(void)
{
  // Compute the eroded intersect mask (m_ErodedIntersectMask) and dilated union mask (m_DilatedUnionMask). 
  ComputeErodedIntersectMask();
  ComputeDilatedUnionMask();
  
  typedef XorImageFilter<TInputMask, TInputMask, TInputMask> XorImageFilterType;
  typename XorImageFilterType::Pointer xorImageFilter = XorImageFilterType::New();
  
  // Calcluate boundray as the XOR region of the eroded intersect mask (m_ErodedIntersectMask)
  // and the dilated union mask (m_DilatedUnionMask). 
  xorImageFilter->SetInput1(m_ErodedIntersectMask);
  xorImageFilter->SetInput2(m_DilatedUnionMask);
  xorImageFilter->Update();
  m_BSIMask = xorImageFilter->GetOutput();
  
  // Take boundary within the dilated sub ROI, if necessary.  
  if (m_SubROIMask.IsNotNull())
  {
    typedef BinariseUsingPaddingImageFilter<TInputMask, TInputMask> BinariseUsingPaddingImageFilterType;
    typename BinariseUsingPaddingImageFilterType::Pointer binariseFilter = BinariseUsingPaddingImageFilterType::New();
    
    std::cerr << "using sub ROI..." << std::endl; 
    // Binarise it for the dilation. 
    binariseFilter->SetInput(m_SubROIMask);
    binariseFilter->SetPaddingValue(m_PaddingValue);
    binariseFilter->Update();

    // Dilate it. 
    typedef MultipleDilateImageFilter<TInputMask> MultipleDilateImageFilterType;
    typename MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
    
    multipleDilateImageFilter->SetNumberOfDilations(this->m_NumberOfSubROIDilation);
    multipleDilateImageFilter->SetInput(binariseFilter->GetOutput());
    multipleDilateImageFilter->Update();
    
    typename TInputMask::Pointer subROIMask =  multipleDilateImageFilter->GetOutput();
    ImageRegionConstIterator<TInputMask>  m_SubROIMaskIterator(subROIMask, 
                                                               subROIMask->GetLargestPossibleRegion());
    ImageRegionIterator<TInputMask>  bsiMaskIterator(m_BSIMask, m_BSIMask->GetLargestPossibleRegion());
    
    for (bsiMaskIterator.GoToBegin(), m_SubROIMaskIterator.GoToBegin(); 
         !bsiMaskIterator.IsAtEnd(); 
         ++bsiMaskIterator, ++m_SubROIMaskIterator)
    {
      if (m_SubROIMaskIterator.Get() == 0)
        bsiMaskIterator.Set(0);
    }
  }
}

template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::IntegrateOverBSIMask(void) throw (ExceptionObject)
{
  m_BoundaryShiftIntegral = 0.0;
  if (m_LowerCutoffValue >=  m_UpperCutoffValue)
  {
    std::cerr << "The lower cut off value must less than the upper cut off value." << std::endl; 
  }
  
  m_BaselineImage->Update();
  m_RepeatImage->Update();
  
  ImageRegionConstIterator<TInputMask>  bsiMaskIterator(m_BSIMask, m_BSIMask->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage>  baselineImageIterator(m_BaselineImage, 
                                                               m_BaselineImage->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage>  repeatImageIterator(m_RepeatImage, 
                                                             m_RepeatImage->GetLargestPossibleRegion());
  
  // Integrate over the m_BSIMask. 
  for (baselineImageIterator.GoToBegin(), repeatImageIterator.GoToBegin(), bsiMaskIterator.GoToBegin();
       !bsiMaskIterator.IsAtEnd();
       ++baselineImageIterator, ++repeatImageIterator, ++bsiMaskIterator)
  {
    if (bsiMaskIterator.Get() != 0)
    {
      double baselineValue = static_cast<double>(baselineImageIterator.Get())/m_BaselineIntensityNormalisationFactor;
      double repeatValue = static_cast<double>(repeatImageIterator.Get())/m_RepeatIntensityNormalisationFactor;
      
      // Clip the intensity values. 
      baselineValue = std::max(baselineValue, m_LowerCutoffValue);
      baselineValue = std::min(baselineValue, m_UpperCutoffValue);
      repeatValue = std::max(repeatValue, m_LowerCutoffValue);
      repeatValue = std::min(repeatValue, m_UpperCutoffValue);
      
      m_BoundaryShiftIntegral += (baselineValue-repeatValue);
    }
  }

  typename TInputImage::SpacingType samplingSpacing = m_RepeatImage->GetSpacing();
  typename TInputImage::SpacingType::ConstIterator samplingSpacingIterator = samplingSpacing.Begin(); 
  double samplingSpacingProduct = 1.0;
  
  // Calculate the product of the sampling space. 
  for (samplingSpacingIterator = samplingSpacing.Begin();
       samplingSpacingIterator != samplingSpacing.End();
       ++samplingSpacingIterator)
  {
    samplingSpacingProduct *= *samplingSpacingIterator;
  }
  
  m_BoundaryShiftIntegral = m_BoundaryShiftIntegral*samplingSpacingProduct/
                            (1000.0*(m_UpperCutoffValue-m_LowerCutoffValue));
}


}

#endif


