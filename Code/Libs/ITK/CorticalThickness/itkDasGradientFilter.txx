/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDasGradientFilter_txx
#define __itkDasGradientFilter_txx

#include "itkDasGradientFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
DasGradientFilter<TScalarType, NDimensions>
::DasGradientFilter()
{

  m_PhiTransformation = VectorImageType::New();
  
  m_ThicknessInterpolator = LinearInterpolatorType::New();
  m_WhiteMatterInterpolator = LinearInterpolatorType::New();
  m_GradientInterpolator = VectorInterpolatorType::New();

  m_TransformImageFilter = TransformImageFilterType::New();
  
  m_GradientFilter = GradientFilterType::New();
  m_GradientFilter->SetUseMillimetreScaling(true);
  m_GradientFilter->SetDivideByTwo(true);
  m_GradientFilter->SetNormalize(false);
  
  m_ReverseGradient = false;
  m_UseGradientTransformedMovingImage = true;
  m_GradientFilterInitialized = false;
  
  niftkitkDebugMacro(<<"DasGradientFilter():Constructed with m_ReverseGradient=" << m_ReverseGradient \
      << ", m_UseGradientTransformedMovingImage=" << m_UseGradientTransformedMovingImage \
      << ", m_GradientFilterInitialized=" << m_GradientFilterInitialized \
      );
}

template <class TScalarType, unsigned int NDimensions>
void
DasGradientFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "ReverseGradient = " << m_ReverseGradient << std::endl;
  os << indent << "UseGradientTransformedMovingImage = " << m_UseGradientTransformedMovingImage << std::endl;
  os << indent << "GradientFilterInitialized = " << m_GradientFilterInitialized << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
DasGradientFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  niftkitkDebugMacro(<< "BeforeThreadedGenerateData():Started");
  
  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 4 inputs, set using the SetInput method.
  if (numberOfInputs != 4)
    {
      itkExceptionMacro(<< "DasGradientFilter should have 4 inputs set using SetInput([0,1,2,3], *image)");
    }
  
  // These are simply set as pointers, so managed externally to this class.
  if (m_PhiTransformation.IsNull())
    {
      itkExceptionMacro(<< "DasGradientFilter: m_PhiTransformation is null");  
    }
  
  InputImageRegionType region;
  for (unsigned int i=0; i<numberOfInputs; i++)
    {
      // Check each input is set.
      InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(i));
      if (!input)
        {
          itkExceptionMacro(<< "Input " << i << " not set!");
        }
        
      // Check they are the same size.
      if (i==0)
        {
          region = input->GetLargestPossibleRegion();
        }
      else if (input->GetLargestPossibleRegion() != region) 
        {
          itkExceptionMacro(<< "All Inputs must have the same dimensions.");
        }
    }  
  
  // Input 0 MUST be the WM image.
  InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  
  if (!m_UseGradientTransformedMovingImage && !m_GradientFilterInitialized)
    {
	  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Updating gradient of WM image");
      m_GradientFilter->SetInput(input);
      m_GradientFilter->UpdateLargestPossibleRegion();
      m_GradientFilterInitialized = true;
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Done");
    }

  // Before calculating gradient, we need to update the transformed moving image.
  if (m_UseGradientTransformedMovingImage)
    {
	  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Calculating transformed moving image.");

      m_TransformImageFilter->SetInput(input);
      m_TransformImageFilter->SetDefaultValue(0);
      m_TransformImageFilter->SetPhiTransformation(m_PhiTransformation);
      m_TransformImageFilter->UpdateLargestPossibleRegion();   
      
      m_GradientFilter->SetInput(m_TransformImageFilter->GetOutput());
      m_GradientFilter->UpdateLargestPossibleRegion();
      
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Done");
    }

  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Finished");
}

template <class TScalarType, unsigned int NDimensions>
void
DasGradientFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, ThreadIdType threadNumber)
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);
  
  // Get Pointers to images.
  typename InputImageType::Pointer wmImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::Pointer wmgmImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typename InputImageType::Pointer thicknessPriorImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(2));

  typename InputImageType::Pointer thicknessImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(3));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionIteratorWithIndex<InputImageType> wmgmIterator(wmgmImage, outputRegionForThread);
  ImageRegionIterator<InputImageType> thicknessPriorIterator(thicknessPriorImage, outputRegionForThread);
  ImageRegionIterator<VectorImageType> phiIterator(m_PhiTransformation, outputRegionForThread);
  ImageRegionIterator<VectorImageType> outputIterator(outputImage, outputRegionForThread);
  
  VectorPixelType zeroPixel; zeroPixel.Fill(0);
  VectorPixelType outputPixel;
  VectorPixelType gradientPixel;
  VectorPixelType phiPixel;
  TScalarType wmPixel;
  TScalarType wmgmPixel;
  TScalarType thicknessPriorPixel;
  TScalarType thicknessPixel;
  Point<TScalarType, Dimension> phiPoint;
  ContinuousIndex<TScalarType, Dimension> continousIndex; 

  if (m_UseGradientTransformedMovingImage)
    {

      m_GradientFilter->Update();
    }
  else
    {
      // Gradient of moving image doesnt change, as is calculated once in "BeforeThreadedGenerateData"
    }
  
  m_GradientInterpolator->SetInputImage(m_GradientFilter->GetOutput());  
  m_WhiteMatterInterpolator->SetInputImage(wmImage);
  m_ThicknessInterpolator->SetInputImage(thicknessImage);
  
  double factor = 1;
  
  if (m_ReverseGradient)
    {
      factor = -1;
    }
    
  for (wmgmIterator.GoToBegin(), 
       thicknessPriorIterator.GoToBegin(), 
       phiIterator.GoToBegin(), 
       outputIterator.GoToBegin();
       !wmgmIterator.IsAtEnd();
       ++wmgmIterator, 
       ++thicknessPriorIterator, 
       ++phiIterator, 
       ++outputIterator)
    {
      wmgmPixel = wmgmIterator.Get();
      thicknessPriorPixel = thicknessPriorIterator.Get();
      phiPixel = phiIterator.Get();
      
      for (unsigned int j = 0; j < Dimension; j++)
        {
          phiPoint[j] = phiPixel[j]; 
        }

      if (wmImage->TransformPhysicalPointToContinuousIndex(phiPoint, continousIndex))
        {
          wmPixel = m_WhiteMatterInterpolator->Evaluate(phiPoint);
          thicknessPixel = m_ThicknessInterpolator->Evaluate(phiPoint);
          gradientPixel = m_GradientInterpolator->Evaluate(phiPoint);
          
          if (thicknessPixel >= thicknessPriorPixel) 
            {
              outputIterator.Set(zeroPixel);
            }
          else
            {
              for (unsigned int j = 0; j < Dimension; j++)
                {
                  outputPixel[j] = (wmPixel - wmgmPixel) * (factor * gradientPixel[j]);  
                }
              
              outputIterator.Set(outputPixel);
            }

        }
      else
        {
          outputIterator.Set(zeroPixel);
        }  
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}

} // end namespace

#endif
