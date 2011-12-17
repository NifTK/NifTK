/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkImageRegistrationFilter_txx
#define __itkImageRegistrationFilter_txx

#include "ConversionUtils.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkLogHelper.h"

namespace itk
{
     
template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter >
ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>
::ImageRegistrationFilter()
{
  m_FinalResampler = ResampleFilterType::New();  
  m_FinalCaster = CastToOutputFilterType::New();
  m_ImageRegistrationFactory = ImageRegistrationFactoryType::New();
  m_AbsImageFilter = AbsImageFilterType::New(); 
  m_Interpolator = m_ImageRegistrationFactory->CreateInterpolator(LINEAR);
  m_MultiResolutionRegistrationMethod = 0;  // must be set by user.  
  m_DoReslicing = true;
  m_IsOutputAbsIntensity = false; 
  m_IsotropicVoxelSize = -1.0; 
  m_ResampledMovingImagePadValue = 0;
  m_ResampledFixedImagePadValue = 0;
  
  niftkitkDebugMacro(<< "ImageRegistrationFilter():With default LINEAR interpolator.");
}
           
template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter >
void 
ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);


  if (! m_FinalResampler.IsNull()) 
    {
      os << indent << "m_FinalResampler:" << std::endl;
      m_FinalResampler.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "m_FinalResampler: NULL" << std::endl;
    }

  if (! m_FinalCaster.IsNull()) 
    {
      os << indent << "m_FinalCaster:" << std::endl;
      m_FinalCaster.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "m_FinalCaster: NULL" << std::endl;
    }

  if (! m_Interpolator.IsNull()) 
    {
      os << indent << "m_Interpolator:" << std::endl;
      m_Interpolator.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "m_Interpolator: NULL" << std::endl;
    }

  os << indent << "m_DoReslicing: " << m_DoReslicing << std::endl;
  os << indent << "m_IsOutputAbsIntensity: " << m_IsOutputAbsIntensity << std::endl;
  os << indent << "m_IsotropicVoxelSize: " << m_IsotropicVoxelSize << std::endl;

  os << indent << "m_ResampledMovingImagePadValue: " << m_ResampledMovingImagePadValue << std::endl;
  os << indent << "m_ResampledFixedImagePadValue: " << m_ResampledFixedImagePadValue << std::endl;
  
  if (!m_MultiResolutionRegistrationMethod.IsNull())
    {
      os << indent << "m_MultiResolutionRegistrationMethod:" << std::endl;
      m_MultiResolutionRegistrationMethod.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "m_MultiResolutionRegistrationMethod: NULL" << std::endl;
    }
}

template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter >
void 
ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>
::Initialize()
{
  niftkitkDebugMacro(<< "Initialize():Started.");
  if (this->m_MultiResolutionRegistrationMethod.GetPointer() == 0)
    {
      itkExceptionMacro(<<"No multi resolution method present");
    }
    
  if (this->m_IsotropicVoxelSize > 0.0)
    {
      niftkitkDebugMacro(<< "Initialize():resampling the input images to isotropic size - " << this->m_IsotropicVoxelSize);
      typename TInputImageType::SpacingType spacing; 
      
      for (unsigned int i = 0; i < Dimension; i++)
        spacing[i] = this->m_IsotropicVoxelSize; 
      m_ResampledFixedImage = ResampleToVoxelSize(this->GetInput(0), m_ResampledFixedImagePadValue, m_ResampleImageInterpolation, spacing); 
      m_ResampledMovingImage = ResampleToVoxelSize(this->m_MovingImage, m_ResampledMovingImagePadValue, m_ResampleImageInterpolation, spacing); 
      if (this->GetInput(1) != NULL)
        m_ResampledFixedMask = ResampleToVoxelSize(this->GetInput(1), 0, m_ResampleMaskInterpolation, spacing); 
      if (!this->m_MovingMask.IsNull())
        m_ResampledMovingMask = ResampleToVoxelSize(this->m_MovingMask, 0, m_ResampleMaskInterpolation, spacing); 
    
      // Resample the images and re-initialise the transform. 
      m_MultiResolutionRegistrationMethod->SetFixedImage(m_ResampledFixedImage);
      m_MultiResolutionRegistrationMethod->SetMovingImage(m_ResampledMovingImage);
      m_MultiResolutionRegistrationMethod->SetFixedMask(m_ResampledFixedMask);
      m_MultiResolutionRegistrationMethod->SetMovingMask(m_ResampledMovingMask);
      niftkitkDebugMacro(<< "Initialize():resampling the input images to isotropic size - " << this->m_IsotropicVoxelSize << " done");
      
      typename FluidDeformableTransformType::Pointer fluidTransform = dynamic_cast<FluidDeformableTransformType*>(this->m_MultiResolutionRegistrationMethod->GetSingleResMethod()->GetTransform()); 

      if (!fluidTransform.IsNull())
      {
        if (this->m_MultiResolutionRegistrationMethod->GetInitialTransformParameters().GetSize() <= 1)
        {
          fluidTransform->Initialize(m_ResampledFixedImage); 
        }
        else
        {
          fluidTransform->SetParameters(this->m_MultiResolutionRegistrationMethod->GetInitialTransformParameters()); 
          fluidTransform->InterpolateNextGrid(m_ResampledFixedImage); 
          this->m_MultiResolutionRegistrationMethod->SetInitialTransformParameters(fluidTransform->GetParameters());
        }
      }
    }    
  else
    {
      m_MultiResolutionRegistrationMethod->SetFixedImage(this->GetInput(0));
      m_MultiResolutionRegistrationMethod->SetMovingImage(this->m_MovingImage);
      m_MultiResolutionRegistrationMethod->SetFixedMask(this->GetInput(1));
      m_MultiResolutionRegistrationMethod->SetMovingMask(this->m_MovingMask);
    }
  niftkitkDebugMacro(<< "Initialize():Finished.");
}

template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter >
void 
ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>
::GenerateData()
{
  niftkitkDebugMacro(<< "Started Registration");
  
  this->Initialize();
  m_MultiResolutionRegistrationMethod->StartRegistration();
    
  niftkitkDebugMacro(<< "Finished Registration");

  typename TInputImageType::ConstPointer fixedImage = this->GetInput(0);
  typename TInputImageType::ConstPointer movingImage = this->m_MovingImage.GetPointer();
  typename TOutputImageType::ConstPointer outputImage = this->GetOutput();
  
  if (m_DoReslicing)
    {
      niftkitkDebugMacro(<< "Started Reslicing");
      
      typename FluidDeformableTransformType::Pointer fluidTransform = dynamic_cast<FluidDeformableTransformType*>(this->m_MultiResolutionRegistrationMethod->GetSingleResMethod()->GetTransform()); 
      if (this->m_IsotropicVoxelSize > 0.0 && !fluidTransform.IsNull())
        {
          fluidTransform->InterpolateNextGrid(fixedImage); 
        }
      
      m_FinalResampler->SetInput(movingImage);
      m_FinalResampler->SetTransform(this->m_MultiResolutionRegistrationMethod->GetSingleResMethod()->GetTransform());

#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
      m_FinalResampler->SetOutputParametersFromImage(const_cast<TInputImageType*>(fixedImage.GetPointer()));
#else
      m_FinalResampler->SetOutputParametersFromImage(fixedImage.GetPointer());  
#endif
      m_FinalResampler->SetDefaultPixelValue(m_ResampledMovingImagePadValue);
      m_FinalResampler->SetInterpolator(m_Interpolator);
      
      if (m_IsOutputAbsIntensity)
        {
          m_AbsImageFilter->SetInput(m_FinalResampler->GetOutput()); 
          m_FinalCaster->SetInput(m_AbsImageFilter->GetOutput());
        }
      else
        {
          m_FinalCaster->SetInput(m_FinalResampler->GetOutput());
        }
      
    } else {

      m_FinalCaster->SetInput(fixedImage);
      
    }

  m_FinalCaster->GraftOutput(this->GetOutput());
  m_FinalCaster->Update();
  this->GraftOutput(m_FinalCaster->GetOutput());

  outputImage = this->GetOutput();

  niftkitkDebugMacro(<< "Output is ready, size:" << outputImage->GetLargestPossibleRegion().GetSize() \
    << ", spacing:" << outputImage->GetSpacing() \
    << ", origin:" << outputImage->GetOrigin() \
    << ", direction:\n" << outputImage->GetDirection());
}


template <typename TInputImageType, typename TOutputImageType, unsigned int Dimension, class TScalarType, typename TDeformationScalar, typename TPyramidFilter >
typename ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>::InputImagePointer
ImageRegistrationFilter<TInputImageType, TOutputImageType, Dimension, TScalarType, TDeformationScalar, TPyramidFilter>
::ResampleToVoxelSize(const TInputImageType* image, const InputPixelType defaultPixelValue, InterpolationTypeEnum interpolation, typename TInputImageType::SpacingType voxelSize)
{
  typedef IdentityTransform< double, Dimension > IdentityTransform;
  typename IdentityTransform::Pointer identityTransform = IdentityTransform::New();
  typename ResampleFilterType::Pointer resampleImageFilter = ResampleFilterType::New(); 
  typename TInputImageType::SpacingType spacing; 
  typename TInputImageType::SpacingType oldSpacing; 
  typename TInputImageType::SizeType regionSize; 
  typename TInputImageType::SizeType oldRegionSize; 
  
  resampleImageFilter->SetInput(image);
  resampleImageFilter->SetTransform(identityTransform);
  resampleImageFilter->SetDefaultPixelValue(defaultPixelValue);
  resampleImageFilter->SetInterpolator(this->m_ImageRegistrationFactory->CreateInterpolator(interpolation));
  resampleImageFilter->SetOutputDirection(image->GetDirection());
  resampleImageFilter->SetOutputOrigin(image->GetOrigin());
  spacing = voxelSize; 
  resampleImageFilter->SetOutputSpacing(spacing);
  oldRegionSize = image->GetLargestPossibleRegion().GetSize(); 
  oldSpacing = image->GetSpacing(); 
  for (unsigned int i = 0; i < Dimension; i++)
  {
    regionSize[i] = static_cast<typename TInputImageType::SizeValueType>(niftk::Round((oldRegionSize[i]*oldSpacing[i])/voxelSize[i])); 
  }
  resampleImageFilter->SetSize(regionSize);
  resampleImageFilter->Update(); 
  
  typename TInputImageType::Pointer resampledImage = resampleImageFilter->GetOutput(); 
  resampledImage->DisconnectPipeline(); 
  
  return resampledImage; 
}



} // end namespace

#endif // __itkImageRegistrationFilter_txx
