/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKBOUNDARYSHIFTINEGRALCALCULATOR_TXX
#define ITKBOUNDARYSHIFTINEGRALCALCULATOR_TXX

#include "itkSegmentationReliabilityCalculator.h"
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
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
::SegmentationReliabilityCalculator()
{
  m_SegmentationReliability = 0.0;
  m_NumberOfErosion = 1;
  m_NumberOfDilation = 1;
  m_PaddingValue = 0;
}

template <class TInputImage, class TInputMask, class TOutputImage>
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
::~SegmentationReliabilityCalculator()
{
}

template <class TInputImage, class TInputMask, class TOutputImage>
void  
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
::Compute()
{
  // Check for same image sizes and voxel sizes. 
  if (!IsSameRegionSize<TInputMask, TInputMask>(this->m_BaselineMask, this->m_RepeatMask))
  {
    itkExceptionMacro("The masks do not have the same number of voxels in each dimension.")
  }
  if (!IsSameVoxelSize<TInputMask, TInputMask>(this->m_BaselineMask, this->m_RepeatMask))
  {
    itkExceptionMacro("The images/masks do not have the same voxel size.")
  }
  
  // Compute BSI mask.
  ComputeBSIMask();
  
  // Compute the BSI value. 
  IntegrateOverBSIMask();
}

template <class TInputImage, class TInputMask, class TOutputImage>
void 
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "m_SegmentationReliability: " << m_SegmentationReliability << std::endl;
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
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
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
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
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
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
}

template <class TInputImage, class TInputMask, class TOutputImage>
void 
SegmentationReliabilityCalculator<TInputImage, TInputMask, TOutputImage>
::IntegrateOverBSIMask(void) throw (ExceptionObject)
{
  double numberOfVoxels = 0.0;
  
  m_SegmentationReliability = 0.0;
  
  ImageRegionConstIterator<TInputMask> bsiMaskIterator(m_BSIMask, m_BSIMask->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage> baselineImageIterator(m_BaselineImage, 
                                                              m_BaselineImage->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage> repeatImageIterator(m_RepeatImage, 
                                                            m_RepeatImage->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputMask> baselineMaskIterator(m_BaselineMask, 
                                                            m_BaselineMask->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputMask> repeatMaskIterator(m_RepeatMask, 
                                                          m_RepeatMask->GetLargestPossibleRegion());
  
  // Integrate over the m_BSIMask. 
  for (baselineImageIterator.GoToBegin(), repeatImageIterator.GoToBegin(), bsiMaskIterator.GoToBegin(), baselineMaskIterator.GoToBegin(), repeatMaskIterator.GoToBegin();
       !bsiMaskIterator.IsAtEnd();
       ++baselineImageIterator, ++repeatImageIterator, ++bsiMaskIterator, ++baselineMaskIterator, ++repeatMaskIterator)
  {
    if (bsiMaskIterator.Get() != 0)
    {
//      typename TInputImage::PixelType baselineValue = baselineImageIterator.Get();
//      typename TInputImage::PixelType repeatValue = repeatImageIterator.Get();
      typename TInputMask::PixelType baselineMaskValue = baselineMaskIterator.Get();
      typename TInputMask::PixelType repeatMaskValue = repeatMaskIterator.Get();
      
      numberOfVoxels++; 
      if (baselineMaskValue == repeatMaskValue)
      {
        m_SegmentationReliability++; 
      }
    }
  }

  m_SegmentationReliability = m_SegmentationReliability/numberOfVoxels;
  
}


}

#endif


