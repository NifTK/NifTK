/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSSDRegistrationForceFilter_txx
#define __itkSSDRegistrationForceFilter_txx

#include "itkSSDRegistrationForceFilter.h"
#include <itkSubtractImageFilter.h>
#include <itkGradientImageFilter.h>
#include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#include <itkCastImageFilter.h>

#include <itkLogHelper.h>

namespace itk {

template< class TFixedImage, class TMovingImage, class TScalar >
void
SSDRegistrationForceFilter< TFixedImage, TMovingImage, TScalar >
::GenerateData()
{
  niftkitkDebugMacro(<<"Generating SSD registration force using Christensen's method...");
  
  typedef GradientImageFilter< TMovingImage, float, float > GradientImageFilterType; 
  typename GradientImageFilterType::Pointer gradientImageFilter = GradientImageFilterType::New(); 
  typedef Image< float, TMovingImage::ImageDimension > FloatImageType;
  typedef CastImageFilter< TMovingImage, FloatImageType > MovingToFloatCastImageFilterType; 
  typename MovingToFloatCastImageFilterType::Pointer movingToFloatCastImageFilter = MovingToFloatCastImageFilterType::New(); 
  typedef CastImageFilter< FloatImageType, TMovingImage > FloatToMovingCastImageFilterType; 
  typename FloatToMovingCastImageFilterType::Pointer floatToMovingCastImageFilter = FloatToMovingCastImageFilterType::New(); 
  
  typedef CurvatureAnisotropicDiffusionImageFilter< FloatImageType, FloatImageType > DiffusionImageFilterType; 
  typename DiffusionImageFilterType::Pointer smoothingImageFilter = DiffusionImageFilterType::New(); 
  
  typename TMovingImage::ConstPointer movingImage = this->GetInput(1); 
  
  // Optional edge-preserve smoothing. 
  if (this->m_Smoothing)
  {
    movingToFloatCastImageFilter->SetInput(movingImage); 
    smoothingImageFilter->SetInput(movingToFloatCastImageFilter->GetOutput()); 
    smoothingImageFilter->SetNumberOfIterations(10); 
    smoothingImageFilter->SetTimeStep(0.0625);
    smoothingImageFilter->SetConductanceParameter(0.5);
    floatToMovingCastImageFilter->SetInput(smoothingImageFilter->GetOutput()); 
    floatToMovingCastImageFilter->Update(); 
    movingImage = floatToMovingCastImageFilter->GetOutput(); 
  }
  
  // Gradient image from the moving image. 
  gradientImageFilter->SetInput(movingImage); 
  gradientImageFilter->Update(); 
  
  // Allocate space for output. 
  this->AllocateOutputs(); 

  // Multiply the subtraction image to the gradient image. 
  ImageRegionConstIteratorWithIndex< TFixedImage > fixedImageIterator(this->GetInput(0), this->GetInput(0)->GetLargestPossibleRegion()); 
  ImageRegionConstIteratorWithIndex< TFixedImage > movingImageIterator(this->GetInput(1), this->GetInput(1)->GetLargestPossibleRegion()); 
  double fixedImageMean = 0.0; 
  double movingImageMean = 0.0; 
  double numberOfVoxels = 0.0; 
  double normalisationFactor = 1.0; 
  
  if (m_IsIntensityNormalised)
  {
    niftkitkDebugMacro(<<"m_IsIntensityNormalised is true: calculating means");
    for (movingImageIterator.GoToBegin(), fixedImageIterator.GoToBegin(); 
        !movingImageIterator.IsAtEnd(); 
        ++movingImageIterator, ++fixedImageIterator)
    {
      // Check if the point is inside the fixed image mask. 
      typename TFixedImage::PointType physicalPoint; 
      this->GetInput(0)->TransformIndexToPhysicalPoint(fixedImageIterator.GetIndex(), physicalPoint); 
      if (this->m_FixedImageMask == NULL || this->m_FixedImageMask->IsInside(physicalPoint))
      {
          fixedImageMean += fixedImageIterator.Get(); 
          movingImageMean += movingImageIterator.Get(); 
          numberOfVoxels++; 
      }
    }
    fixedImageMean /= numberOfVoxels; 
    movingImageMean /= numberOfVoxels; 
    normalisationFactor = fixedImageMean/movingImageMean; 
    niftkitkDebugMacro(<<"Fixed image mean=" << fixedImageMean << ", moving image mean=" << movingImageMean << ",normalisation factor=" << normalisationFactor);
  }
  
  ImageRegionConstIterator< typename GradientImageFilterType::OutputImageType > gradientImageIterator(gradientImageFilter->GetOutput(), gradientImageFilter->GetOutput()->GetLargestPossibleRegion()); 
  ImageRegionIterator< OutputImageType > outputImageIterator(this->GetOutput(0), this->GetOutput(0)->GetLargestPossibleRegion()); 
  
  for (gradientImageIterator.GoToBegin(), movingImageIterator.GoToBegin(), outputImageIterator.GoToBegin(), fixedImageIterator.GoToBegin(); 
       !gradientImageIterator.IsAtEnd(); 
       ++gradientImageIterator, ++movingImageIterator, ++outputImageIterator, ++fixedImageIterator)
  {
    typename OutputImageType::PixelType outputValue; 
    
    // Check if the point is inside the fixed image mask. 
    typename TFixedImage::PointType physicalPoint; 
    this->GetInput(0)->TransformIndexToPhysicalPoint(fixedImageIterator.GetIndex(), physicalPoint); 
    if (this->m_FixedImageMask != NULL && !this->m_FixedImageMask->IsInside(physicalPoint))
    {
      for (unsigned int i = 0; i < TFixedImage::ImageDimension; i++)
        outputValue[i] = 0; 
      outputImageIterator.Set(outputValue); 
      continue;       
    }
    
    double diff = (movingImageIterator.Get()*normalisationFactor) - (fixedImageIterator.Get()); 
    
    for (unsigned int i = 0; i < TFixedImage::ImageDimension; i++)
      outputValue[i] = gradientImageIterator.Get()[i] * diff; 
    outputImageIterator.Set(outputValue); 
  }
  niftkitkDebugMacro(<<"Generating SSD registration force using Christensen's method...done.");
}





}

#endif

