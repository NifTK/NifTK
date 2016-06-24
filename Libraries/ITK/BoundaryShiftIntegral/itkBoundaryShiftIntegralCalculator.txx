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
#include <itkImageRegionConstIterator.h>
#include <itkXorImageFilter.h>
#include <itkImageFileWriter.h>
#include "itkBinariseUsingPaddingImageFilter.h"
#include "itkBasicImageComparisonFunctions.h"
#include "itkMultipleDilateImageFilter.h"
#include "itkMultipleErodeImageFilter.h"
#include "itkCastImageFilter.h"
#include <algorithm>
#include <iostream>

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
  m_ProbabilisticBSI=0;
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
	
  // Compute BSI mask depending on parameters
  if(m_ProbabilisticBSI==0) {
	typedef itk::CastImageFilter< TInputMask, IntImageType > CastToIntFilterType;
	typename CastToIntFilterType::Pointer castToIntFilter1 = CastToIntFilterType::New();
		
	castToIntFilter1->SetInput(m_BaselineMask);
	castToIntFilter1->Update();
	m_BaselineMaskInt=castToIntFilter1->GetOutput();
	
	typename CastToIntFilterType::Pointer castToIntFilter2 = CastToIntFilterType::New();
	castToIntFilter2->SetInput(m_RepeatMask);
	castToIntFilter2->Update();
	m_RepeatMaskInt=castToIntFilter2->GetOutput();
	
	ComputeBSIMask();
  }
  else if(m_ProbabilisticBSI==1) {
	// If 1 we compute PXOR
	ComputeGBSIMask();
  } 
  else if(m_ProbabilisticBSI==2 || m_ProbabilisticBSI==3) {
	// If 2 we compute pBSI1, if 3 pBSIgamma
	ComputeLedigMask();
  }
  // Otherwhise we don't compute xor because user specifies it

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
::ComputePORandPANDMaskLedig(void)
{

  m_PAND = TInputMask::New() ;
  m_PAND->SetRegions(m_BaselineMask->GetLargestPossibleRegion() ) ;
  m_PAND->SetOrigin(m_BaselineMask->GetOrigin()); 
  m_PAND->SetSpacing(m_BaselineMask->GetSpacing()); 
  m_PAND->SetDirection(m_BaselineMask->GetDirection()); 
  m_PAND->Allocate();

  m_POR = TInputMask::New() ;
  m_POR->SetRegions(m_BaselineMask->GetLargestPossibleRegion() ) ;
  m_POR->SetOrigin(m_BaselineMask->GetOrigin()); 
  m_POR->SetSpacing(m_BaselineMask->GetSpacing()); 
  m_POR->SetDirection(m_BaselineMask->GetDirection()); 
  m_POR->Allocate();

  typedef itk::ImageRegionIterator< TInputMask > ImageIteratorType ;
  ImageIteratorType baseMask_iter( m_BaselineMask,
                             m_BaselineMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType repeatMask_iter( m_RepeatMask,
                             m_RepeatMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType m_POR_iter( m_POR,
                             m_POR->GetLargestPossibleRegion() ) ;
  ImageIteratorType m_PAND_iter( m_PAND,
                             m_PAND->GetLargestPossibleRegion() ) ;
	     
  for (m_POR_iter.GoToBegin(), m_PAND_iter.GoToBegin(), baseMask_iter.GoToBegin(), repeatMask_iter.GoToBegin();
       !baseMask_iter.IsAtEnd();
       ++m_POR_iter, ++m_PAND_iter, ++baseMask_iter, ++repeatMask_iter)
	{
		double A=static_cast<double>(baseMask_iter.Get());
		double B=static_cast<double>(repeatMask_iter.Get());
		double fuzzy_or=A>B?A:B;
		double fuzzy_and=A>B?B:A;
		int fuzzy_AND=fuzzy_and>=0.95?1:0;
		int fuzzy_OR=fuzzy_or>=0.9?1:0;
		m_POR_iter.Set(fuzzy_OR);
		m_PAND_iter.Set(fuzzy_AND);
	}

 if(this->m_NumberOfErosion>0) {	
	  typedef MultipleErodeImageFilter<TInputMask> MultipleErodeImageFilterType;
	  typename MultipleErodeImageFilterType::Pointer multipleErodeImageFilter = MultipleErodeImageFilterType::New();
  
	  // Erode multiple times. 
	  multipleErodeImageFilter->SetNumberOfErosions(this->m_NumberOfErosion);
	  multipleErodeImageFilter->SetInput(m_PAND);
	  multipleErodeImageFilter->Update();
	  m_PAND = multipleErodeImageFilter->GetOutput();
  }

  if(this->m_NumberOfDilation>0) {	
	  typedef MultipleDilateImageFilter<TInputMask> MultipleDilateImageFilterType;
	  typename MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  
	  // Dilate multiple times. 
	  multipleDilateImageFilter->SetNumberOfDilations(this->m_NumberOfDilation);
	  multipleDilateImageFilter->SetInput(m_POR);
	  multipleDilateImageFilter->Update();
	  m_POR = multipleDilateImageFilter->GetOutput();
  }
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeLedigMask(void)
{
  ComputePORandPANDMaskLedig();

  m_BSIMask = TInputMask::New() ;
  m_BSIMask->SetRegions(m_BaselineMask->GetLargestPossibleRegion() ) ;
  m_BSIMask->SetOrigin(m_BaselineMask->GetOrigin()); 
  m_BSIMask->SetSpacing(m_BaselineMask->GetSpacing()); 
  m_BSIMask->SetDirection(m_BaselineMask->GetDirection()); 
  m_BSIMask->Allocate();


  typedef itk::ImageRegionIterator< TInputMask > ImageIteratorType ;
  ImageIteratorType baseMask_iter( m_BaselineMask,
                             m_BaselineMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType repeatMask_iter( m_RepeatMask,
                             m_RepeatMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType bsiMaskIterator( m_BSIMask,
                             m_BSIMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType m_PAND_iter( m_PAND,
                             m_PAND->GetLargestPossibleRegion() ) ;
  ImageIteratorType m_POR_iter( m_POR,
                             m_POR->GetLargestPossibleRegion() ) ;
	     
  for (bsiMaskIterator.GoToBegin(), m_PAND_iter.GoToBegin(), m_POR_iter.GoToBegin(), baseMask_iter.GoToBegin(), repeatMask_iter.GoToBegin();
       !m_PAND_iter.IsAtEnd();
       ++bsiMaskIterator, ++m_PAND_iter, ++m_POR_iter, ++baseMask_iter, ++repeatMask_iter)
	{
		double A=static_cast<double>(m_PAND_iter.Get());
		double B=static_cast<double>(m_POR_iter.Get());
		double pXOR=(B-A);
		// We apply pBSI gamma, with gamma=0.5 like in Ledig et al. 2012
		if(m_ProbabilisticBSI==3 && pXOR!=0) {
			pXOR=std::max(0.5,std::max((double)baseMask_iter.Get(),(double)repeatMask_iter.Get()));
		}
		bsiMaskIterator.Set(pXOR);
	}
}

// We compute pXOR mask directly from two time-points mask
template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeGBSIMask(void)
{
  m_BSIMask = TInputMask::New() ;
  m_BSIMask->SetRegions(m_BaselineMask->GetLargestPossibleRegion() ) ;
  m_BSIMask->SetOrigin(m_BaselineMask->GetOrigin()); 
  m_BSIMask->SetSpacing(m_BaselineMask->GetSpacing()); 
  m_BSIMask->SetDirection(m_BaselineMask->GetDirection()); 
  m_BSIMask->Allocate();

  typedef itk::ImageRegionIterator< TInputMask > ImageIteratorType ;
  ImageIteratorType bsiMaskIterator( m_BSIMask,
                             m_BSIMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType baseMask_iter( m_BaselineMask,
                             m_BaselineMask->GetLargestPossibleRegion() ) ;
  ImageIteratorType repeatMask_iter( m_RepeatMask,
                             m_RepeatMask->GetLargestPossibleRegion() ) ;
  int p=0;
  double mean=0.0;			     
  for (bsiMaskIterator.GoToBegin(), baseMask_iter.GoToBegin(), repeatMask_iter.GoToBegin();
       !baseMask_iter.IsAtEnd();
       ++bsiMaskIterator, ++baseMask_iter, ++repeatMask_iter)
	{
		double A=static_cast<double>(baseMask_iter.Get());
		double B=static_cast<double>(repeatMask_iter.Get());
		double p1=(A*(1-B));
		double p2=((1-A)*B);
		double pXOR=(p1+p2-p1*p2);
		if(pXOR>1) pXOR=1;
		bsiMaskIterator.Set(pXOR);
		mean+=pXOR;
		if(pXOR!=0) {
			p++;
		}
	}
  mean=mean/p;
  for (bsiMaskIterator.GoToBegin();
       !bsiMaskIterator.IsAtEnd();
       ++bsiMaskIterator)
	{
		double pXOR=bsiMaskIterator.Get();
		double xorval=pXOR<(mean)?pXOR/(mean):1.0;
	        bsiMaskIterator.Set(xorval);
	}
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
BoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::ComputeErodedIntersectMask(void)
{
  
  typedef BinaryIntersectWithPaddingImageFilter<IntImageType, IntImageType> IntersectFilterType;
  typename IntersectFilterType::Pointer intersectFilter = IntersectFilterType::New();
  
  // Intersect the two masks.
  intersectFilter->SetInput1(m_BaselineMaskInt);
  intersectFilter->SetInput2(m_RepeatMaskInt);
  intersectFilter->SetPaddingValue(m_PaddingValue);
  
  typedef MultipleErodeImageFilter<IntImageType> MultipleErodeImageFilterType;
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
  typedef BinaryUnionWithPaddingImageFilter<IntImageType, IntImageType> UnionFilterType;
  typename UnionFilterType::Pointer unionFilter = UnionFilterType::New();
  
  // Union the two masks.
  unionFilter->SetInput1(m_BaselineMaskInt);
  unionFilter->SetInput2(m_RepeatMaskInt);
  unionFilter->SetPaddingValue(m_PaddingValue);
  
  typedef MultipleDilateImageFilter<IntImageType> MultipleDilateImageFilterType;
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
    
  typedef XorImageFilter<IntImageType, IntImageType, IntImageType> XorImageFilterType;
  typename XorImageFilterType::Pointer xorImageFilter = XorImageFilterType::New();
  
  // Calculate boundray as the XOR region of the eroded intersect mask (m_ErodedIntersectMask)
  // and the dilated union mask (m_DilatedUnionMask). 
  xorImageFilter->SetInput1(m_ErodedIntersectMask);
  xorImageFilter->SetInput2(m_DilatedUnionMask);
  xorImageFilter->Update();
  //m_BSIMask = xorImageFilter->GetOutput();
  
  typedef itk::CastImageFilter<IntImageType,TInputMask> CastToDoubleFilterType;
  typename CastToDoubleFilterType::Pointer castToDoubleFilter = CastToDoubleFilterType::New();
		
  castToDoubleFilter->SetInput(xorImageFilter->GetOutput());
  castToDoubleFilter->Update();
  m_BSIMask=castToDoubleFilter->GetOutput();
  
  
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
  
  double valueBSI=0;
  TInputImagePointer outputImage = TInputImage::New() ;
  outputImage->SetRegions( m_BaselineImage->GetLargestPossibleRegion() ) ;
  outputImage->SetOrigin(m_BaselineImage->GetOrigin()); 
  outputImage->SetSpacing(m_BaselineImage->GetSpacing()); 
  outputImage->SetDirection(m_BaselineImage->GetDirection()); 
  outputImage->Allocate() ;

  TInputImagePointer xoroutputImage = TInputImage::New() ;
  xoroutputImage->SetRegions( m_BaselineImage->GetLargestPossibleRegion() ) ;
  xoroutputImage->SetOrigin(m_BaselineImage->GetOrigin()); 
  xoroutputImage->SetSpacing(m_BaselineImage->GetSpacing()); 
  xoroutputImage->SetDirection(m_BaselineImage->GetDirection()); 
  xoroutputImage->Allocate() ;
  
  typedef itk::ImageRegionIterator< TInputImage > ImageIteratorType ;
  ImageIteratorType io_iter( outputImage,
                             outputImage->GetLargestPossibleRegion() ) ;
  ImageIteratorType xor_iter( xoroutputImage,
                             xoroutputImage->GetLargestPossibleRegion() ) ;
  xor_iter.GoToBegin();
  io_iter.GoToBegin();
  // Integrate over the m_BSIMask. 
  for (baselineImageIterator.GoToBegin(), repeatImageIterator.GoToBegin(), bsiMaskIterator.GoToBegin();
       !bsiMaskIterator.IsAtEnd();
       ++baselineImageIterator, ++repeatImageIterator, ++bsiMaskIterator,  ++xor_iter, ++io_iter)
  {
    
    if (bsiMaskIterator.Get() != 0)
    {
      double baselineValue = static_cast<double>(baselineImageIterator.Get())/m_BaselineIntensityNormalisationFactor;
      double repeatValue = static_cast<double>(repeatImageIterator.Get())/m_RepeatIntensityNormalisationFactor;
      double mask=static_cast<double>(bsiMaskIterator.Get());
      
      // Clip the intensity values. 
      baselineValue = std::max(baselineValue, m_LowerCutoffValue);
      baselineValue = std::min(baselineValue, m_UpperCutoffValue);
      repeatValue = std::max(repeatValue, m_LowerCutoffValue);
      repeatValue = std::min(repeatValue, m_UpperCutoffValue);
      
      valueBSI=mask*(baselineValue-repeatValue)/(m_UpperCutoffValue-m_LowerCutoffValue);
      
      m_BoundaryShiftIntegral += mask*(baselineValue-repeatValue);
      xor_iter.Set(mask);
      io_iter.Set(valueBSI);
    }
    else { 
	xor_iter.Set(0);
	io_iter.Set(0);
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


  m_BSIMapSIENAStyle=outputImage;
  m_XORMap=xoroutputImage;
}


}

#endif


