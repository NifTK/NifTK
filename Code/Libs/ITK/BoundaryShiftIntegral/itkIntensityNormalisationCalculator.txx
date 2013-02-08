/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKINTENSITYNORMALISATIONCALCULATOR_TXX_
#define ITKINTENSITYNORMALISATIONCALCULATOR_TXX_

#include "itkIntensityNormalisationCalculator.h"
#include "itkBinaryIntersectWithPaddingImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageFileWriter.h"
#include "itkBasicImageComparisonFunctions.h"
#include <iostream>

namespace itk
{

template<class TInputImage, class TInputMask>
IntensityNormalisationCalculator<TInputImage, TInputMask>
::IntensityNormalisationCalculator()
{
  m_NormalisationMean1 = 0.0;
  m_NormalisationMean2 = 0.0;
  m_PaddingValue = 0;
}


template<class TInputImage, class TInputMask>
void  
IntensityNormalisationCalculator<TInputImage, TInputMask>
::Compute()
{
  // Check for same image sizes and voxel sizes. 
  if (!IsSameRegionSize<TInputImage, TInputMask>(this->m_InputImage1, this->m_InputMask1) ||
      !IsSameRegionSize<TInputMask, TInputImage>(this->m_InputMask1, this->m_InputImage2) ||
      !IsSameRegionSize<TInputImage, TInputMask>(this->m_InputImage2, this->m_InputMask2))
  {
    itkExceptionMacro("The images/masks do not have the same number of voxels in each dimension.")
  }
  if (!IsSameVoxelSize<TInputImage, TInputMask>(this->m_InputImage1, this->m_InputMask1) ||
      !IsSameVoxelSize<TInputMask, TInputImage>(this->m_InputMask1, this->m_InputImage2) ||
      !IsSameVoxelSize<TInputImage, TInputMask>(this->m_InputImage2, this->m_InputMask2))
  {
    itkExceptionMacro("The images/masks do not have the same voxel size.")
  }
  
  
  typedef BinaryIntersectWithPaddingImageFilter<TInputMask, TInputMask> IntersectFilterType;
  typename IntersectFilterType::Pointer intersectFilter = IntersectFilterType::New();
  
  // Intersect the two masks.
  intersectFilter->SetInput1(m_InputMask1);
  intersectFilter->SetInput2(m_InputMask2);
  intersectFilter->SetPaddingValue(m_PaddingValue);
  
  typedef itk::BinaryCrossStructuringElement<typename TInputMask::PixelType, TInputMask::ImageDimension> 
    StructuringElementType;
  typedef itk::BinaryErodeImageFilter<TInputMask, TInputMask, StructuringElementType> 
    ErodeImageFilterType;
  StructuringElementType structuringElement; 
  typename ErodeImageFilterType::Pointer erodeImageFilter = ErodeImageFilterType::New();
  
  // Erode the intersection once. 
  structuringElement.SetRadius(1);
  structuringElement.CreateStructuringElement();
  erodeImageFilter->SetInput(intersectFilter->GetOutput());
  erodeImageFilter->SetKernel(structuringElement);
  erodeImageFilter->SetErodeValue(1);
  erodeImageFilter->SetBackgroundValue(0);
  erodeImageFilter->SetBoundaryToForeground(false);
  erodeImageFilter->Update();
  m_InputImage1->Update();
  m_InputImage2->Update();
  
  ImageRegionConstIterator<TInputImage>  inputImageIterator1(m_InputImage1, m_InputImage1->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage>  inputImageIterator2(m_InputImage2, m_InputImage2->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputMask>  maskIterator(erodeImageFilter->GetOutput(), erodeImageFilter->GetOutput()->GetLargestPossibleRegion());
  unsigned int numberOfPixels = 0;
  
  // Calculate the means in the two images within the eroded intersect region. 
  m_NormalisationMean1 = 0.0;
  m_NormalisationMean2 = 0.0;
  maskIterator.GoToBegin();
  inputImageIterator1.GoToBegin();
  inputImageIterator2.GoToBegin();
  while (!maskIterator.IsAtEnd())
  {
    const typename TInputMask::PixelType maskValue = maskIterator.Get();
    
    if (maskValue != 0) 
    {
      m_NormalisationMean1 += static_cast<double>(inputImageIterator1.Get());
      m_NormalisationMean2 += static_cast<double>(inputImageIterator2.Get());
      numberOfPixels++;
    }
    ++maskIterator;
    ++inputImageIterator1;
    ++inputImageIterator2;
  }

  if (numberOfPixels > 0)
  {
    m_NormalisationMean1 = m_NormalisationMean1/static_cast<double>(numberOfPixels);
    m_NormalisationMean2 = m_NormalisationMean2/static_cast<double>(numberOfPixels);
  }
}

template<class TInputImage, class TInputMask>
void 
IntensityNormalisationCalculator<TInputImage, TInputMask>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "m_NormalisationMean1: " << m_NormalisationMean1 << std::endl;
  os << indent << "m_NormalisationMean2: " << m_NormalisationMean2 << std::endl;
}

} // end namespace itk

#endif
