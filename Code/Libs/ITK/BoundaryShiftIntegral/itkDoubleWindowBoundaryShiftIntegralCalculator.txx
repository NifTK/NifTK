/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDOUBLEWINDOWBOUNDARYSHIFTINEGRALCALCULATOR_TXX
#define ITKDOUBLEWINDOWBOUNDARYSHIFTINEGRALCALCULATOR_TXX

#include "itkDoubleWindowBoundaryShiftIntegralCalculator.h"

namespace itk
{
  
template <class TInputImage, class TInputMask, class TOutputImage>
DoubleWindowBoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::DoubleWindowBoundaryShiftIntegralCalculator()
{
  this->m_SecondUpperCutoffValue = 1.10;
  this->m_SecondLowerCutoffValue = 0.90;
  this->m_FirstBoundaryShiftIntegral = 0.0; 
  this->m_SecondBoundaryShiftIntegral = 0.0;
  this->m_MinSecondWindowWidth = -1.0; 
}


template <class TInputImage, class TInputMask, class TOutputImage>
void 
DoubleWindowBoundaryShiftIntegralCalculator<TInputImage, TInputMask, TOutputImage>
::IntegrateOverBSIMask(void) throw (ExceptionObject)
{
  // Check window width. 
  double secondLowerCutoffValue = this->m_SecondLowerCutoffValue; 
  double secondUpperCutoffValue = this->m_SecondUpperCutoffValue; 
  if (this->m_MinSecondWindowWidth > 0.0)
  {
    // Re-adjust the window width. 
    if (this->m_SecondUpperCutoffValue - this->m_SecondLowerCutoffValue < this->m_MinSecondWindowWidth)
    {
      double centre = (this->m_SecondUpperCutoffValue + this->m_SecondLowerCutoffValue)/2.0; 
      secondLowerCutoffValue = centre - this->m_MinSecondWindowWidth/2.0; 
      secondUpperCutoffValue = centre + this->m_MinSecondWindowWidth/2.0; 
    }
  }
  
  // Allocate memory for the BSI maps. 
  this->m_BSIMap = TOutputImage::New(); 
  this->m_BSIMap->SetRegions(this->m_BaselineImage->GetLargestPossibleRegion()); 
  this->m_BSIMap->SetOrigin(this->m_BaselineImage->GetOrigin()); 
  this->m_BSIMap->SetSpacing(this->m_BaselineImage->GetSpacing()); 
  this->m_BSIMap->SetDirection(this->m_BaselineImage->GetDirection()); 
  this->m_BSIMap->Allocate(); 
  this->m_BSIMap->FillBuffer(1000.); 
  this->m_SecondBSIMap = TOutputImage::New(); 
  this->m_SecondBSIMap->SetRegions(this->m_BaselineImage->GetLargestPossibleRegion()); 
  this->m_SecondBSIMap->SetOrigin(this->m_BaselineImage->GetOrigin()); 
  this->m_SecondBSIMap->SetSpacing(this->m_BaselineImage->GetSpacing()); 
  this->m_SecondBSIMap->SetDirection(this->m_BaselineImage->GetDirection()); 
  this->m_SecondBSIMap->Allocate(); 
  this->m_SecondBSIMap->FillBuffer(1000.); 
  
  this->m_BoundaryShiftIntegral = 0.0;
  this->m_FirstBoundaryShiftIntegral = 0.0; 
  this->m_SecondBoundaryShiftIntegral = 0.0; 
  if (this->m_LowerCutoffValue >=  this->m_UpperCutoffValue)
  {
    itkExceptionMacro("The first lower cut off value must less than the fist upper cut off value.")
  }
  if (this->m_UpperCutoffValue >= secondLowerCutoffValue)
  {
    itkExceptionMacro("The first uppper cut off value must less than the second lower cut off value.")
  }
  
  this->m_BaselineImage->Update();
  this->m_RepeatImage->Update();
  
  ImageRegionConstIterator<TInputMask>  bsiMaskIterator(this->m_BSIMask, this->m_BSIMask->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage>  baselineImageIterator(this->m_BaselineImage, 
                                                               this->m_BaselineImage->GetLargestPossibleRegion());
  ImageRegionConstIterator<TInputImage>  repeatImageIterator(this->m_RepeatImage, 
                                                             this->m_RepeatImage->GetLargestPossibleRegion());
  ImageRegionIterator<TOutputImage> bsiMapIterator(this->m_BSIMap, this->m_BSIMap->GetLargestPossibleRegion()); 
  ImageRegionIterator<TOutputImage> secondBSIMapIterator(this->m_SecondBSIMap, this->m_SecondBSIMap->GetLargestPossibleRegion()); 
  
  // Integrate over the m_BSIMask or over the weight image. 
  for (baselineImageIterator.GoToBegin(), repeatImageIterator.GoToBegin(), bsiMaskIterator.GoToBegin(), bsiMapIterator.GoToBegin(), secondBSIMapIterator.GoToBegin();
       !bsiMaskIterator.IsAtEnd();
       ++baselineImageIterator, ++repeatImageIterator, ++bsiMaskIterator, ++bsiMapIterator, ++secondBSIMapIterator)
  {
    double weight = 0.0; 
    
    if (this->m_WeightImage.IsNull())
    {
      if (bsiMaskIterator.Get() != 0)
        weight = 1.0; 
    }
    else
    {
      weight = this->m_WeightImage->GetPixel(baselineImageIterator.GetIndex()); 
    }
      
    double baselineValue = static_cast<double>(baselineImageIterator.Get())/this->m_BaselineIntensityNormalisationFactor;
    double repeatValue = static_cast<double>(repeatImageIterator.Get())/this->m_RepeatIntensityNormalisationFactor;
    
    // The second window deals with intensity changes above the first upper cut off value. 
    if (baselineValue > this->m_UpperCutoffValue && repeatValue > this->m_UpperCutoffValue)
    {
      // The intensity around the second could be noisy - in this case - don't use it. 
      if (secondLowerCutoffValue < secondUpperCutoffValue)
      {
        // Clip the intensity values. 
        baselineValue = std::max(baselineValue, secondLowerCutoffValue);
        baselineValue = std::min(baselineValue, secondUpperCutoffValue);
        repeatValue = std::max(repeatValue, secondLowerCutoffValue);
        repeatValue = std::min(repeatValue, secondUpperCutoffValue);
        
        double bsi = weight*(baselineValue-repeatValue)/(secondUpperCutoffValue-secondLowerCutoffValue); 
        this->m_SecondBoundaryShiftIntegral += bsi;
        secondBSIMapIterator.Set(static_cast<typename TOutputImage::PixelType>((bsi+1)*1000.)); 
      }
    }
    else
    {
      // Clip the intensity values. 
      baselineValue = std::max(baselineValue, this->m_LowerCutoffValue);
      baselineValue = std::min(baselineValue, this->m_UpperCutoffValue);
      repeatValue = std::max(repeatValue, this->m_LowerCutoffValue);
      repeatValue = std::min(repeatValue, this->m_UpperCutoffValue);
      
      double bsi = weight*(baselineValue-repeatValue)/(this->m_UpperCutoffValue-this->m_LowerCutoffValue); 
      this->m_FirstBoundaryShiftIntegral += bsi;
      bsiMapIterator.Set(static_cast<typename TOutputImage::PixelType>((bsi+1)*1000.)); 
    }
  }

  typename TInputImage::SpacingType samplingSpacing = this->m_RepeatImage->GetSpacing();
  typename TInputImage::SpacingType::ConstIterator samplingSpacingIterator = samplingSpacing.Begin(); 
  double samplingSpacingProduct = 1.0;
  
  // Calculate the product of the sampling space. 
  for (samplingSpacingIterator = samplingSpacing.Begin();
       samplingSpacingIterator != samplingSpacing.End();
       ++samplingSpacingIterator)
  {
    samplingSpacingProduct *= *samplingSpacingIterator;
  }
  
  this->m_FirstBoundaryShiftIntegral = this->m_FirstBoundaryShiftIntegral*samplingSpacingProduct/(1000.0);
  this->m_SecondBoundaryShiftIntegral = this->m_SecondBoundaryShiftIntegral*samplingSpacingProduct/(1000.0);
  
  this->m_BoundaryShiftIntegral = this->m_FirstBoundaryShiftIntegral-this->m_SecondBoundaryShiftIntegral; 
}

  
  
}

#endif
