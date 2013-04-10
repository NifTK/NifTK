/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkMaskedImageRegistrationMethod_txx
#define _itkMaskedImageRegistrationMethod_txx

#include "itkLogHelper.h"
#include "itkMaskedImageRegistrationMethod.h"


namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType >
MaskedImageRegistrationMethod<TInputImageType>
::MaskedImageRegistrationMethod()
: SingleResolutionImageRegistrationMethod<TInputImageType,TInputImageType>()
{

  m_Sigma = 0;

  m_FixedMask = 0;
  m_MovingMask = 0;
  m_UseFixedMask = false;
  m_UseMovingMask = false;
  m_MaskImageDirectly = false;
  m_NumberOfDilations = 0;

  m_ThresholdFixedMask = false;
  m_FixedMaskMinimum = 1;
  m_FixedMaskMaximum = std::numeric_limits<InputImagePixelType>::max();
  m_ThresholdMovingMask = false;
  m_MovingMaskMinimum = 1;
  m_MovingMaskMaximum = std::numeric_limits<InputImagePixelType>::max();
  
  m_FixedRescaler = RescaleFilterType::New();
  m_MovingRescaler = RescaleFilterType::New();

  m_FixedSmoother = SmoothingFilterType::New();
  m_MovingSmoother = SmoothingFilterType::New();
  
  m_FixedMaskThresholder = ThresholdFilterType::New();
  m_MovingMaskThresholder = ThresholdFilterType::New();
  
  m_FixedMaskDilater = DilateMaskFilterType::New();
  m_MovingMaskDilater = DilateMaskFilterType::New();
  
  m_FixedImageMuliplier = MultiplyFilterType::New();
  m_MovingImageMultiplier = MultiplyFilterType::New();

  m_FixedMaskCaster = CastToMaskImageTypeFilterType::New();
  m_MovingMaskCaster = CastToMaskImageTypeFilterType::New();
  
  m_FixedMasker = MaskFilterType::New();
  m_MovingMasker = MaskFilterType::New();
  
  m_FixedImageCopy = InputImageType::New();
  m_MovingImageCopy = InputImageType::New();

  m_RescaleFixedImage = false;
  m_RescaleFixedMinimum = 0;
  m_RescaleFixedMaximum = 255;
  m_RescaleMovingImage = false;
  m_RescaleMovingMinimum = 0;
  m_RescaleMovingMaximum = 255;   
  m_RescaleFixedBoundaryValue = m_FixedRescaler->GetOutputBoundaryValue();
  m_RescaleFixedLowerThreshold = m_FixedRescaler->GetInputLowerThreshold();
  m_RescaleFixedUpperThreshold = m_FixedRescaler->GetInputUpperThreshold();
  m_RescaleMovingBoundaryValue = m_MovingRescaler->GetOutputBoundaryValue();
  m_RescaleMovingLowerThreshold = m_MovingRescaler->GetInputLowerThreshold();
  m_RescaleMovingUpperThreshold = m_MovingRescaler->GetInputUpperThreshold();

  niftkitkDebugMacro(<<"MaskedImageRegistrationMethod():Constructed");
}

/*
 * PrintSelf
 */
template < typename TInputImageType  >
void
MaskedImageRegistrationMethod<TInputImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  
  os << indent << "UseFixedMask: " << m_UseFixedMask << std::endl;
  os << indent << "UseMovingMask: " << m_UseMovingMask << std::endl;
  os << indent << "MaskImageDirectly: " << m_MaskImageDirectly << std::endl;
  os << indent << "NumberOfDilations: " << m_NumberOfDilations << std::endl;
  os << indent << "RescaleFixedImage: " << m_RescaleFixedImage << std::endl;
  os << indent << "RescaleFixedMinimum: " << m_RescaleFixedMinimum << std::endl;
  os << indent << "RescaleFixedMaximum: " << m_RescaleFixedMaximum << std::endl;
  os << indent << "RescaleFixedBoundaryValue: " << m_RescaleFixedBoundaryValue << std::endl;
  os << indent << "RescaleFixedLowerThreshold: " << m_RescaleFixedLowerThreshold << std::endl;
  os << indent << "RescaleFixedUpperThreshold: " << m_RescaleFixedUpperThreshold << std::endl;
  os << indent << "RescaleMovingImage: " << m_RescaleMovingImage << std::endl;
  os << indent << "RescaleMovingMinimum: " << m_RescaleMovingMinimum << std::endl;
  os << indent << "RescaleMovingMaximum: " << m_RescaleMovingMaximum << std::endl;
  os << indent << "RescaleMovingBoundaryValue: " << m_RescaleMovingBoundaryValue << std::endl;
  os << indent << "RescaleMovingLowerThreshold: " << m_RescaleMovingLowerThreshold << std::endl;
  os << indent << "RescaleMovingUpperThreshold: " << m_RescaleMovingUpperThreshold << std::endl;
  os << indent << "ThresholdFixedMask: " << m_ThresholdFixedMask << std::endl;
  os << indent << "FixedMaskMinimum: " << m_FixedMaskMinimum << std::endl;
  os << indent << "FixedMaskMaximum: " << m_FixedMaskMaximum << std::endl;
  os << indent << "ThresholdMovingMask: " << m_ThresholdMovingMask << std::endl;
  os << indent << "MovingMaskMinimum: " << m_MovingMaskMinimum << std::endl;
  os << indent << "MovingMaskMaximum: " << m_MovingMaskMaximum << std::endl;
  os << indent << "Sigma: " << m_Sigma << std::endl;  
}

template < typename TInputImageType >
void 
MaskedImageRegistrationMethod<TInputImageType>
::CopyImage(const InputImageType* source, InputImageType* target)
{
  
  if (target->GetLargestPossibleRegion().GetSize() != source->GetLargestPossibleRegion().GetSize())
    {
      InputImageRegionType    region    = source->GetLargestPossibleRegion();
      InputImageOriginType    origin    = source->GetOrigin();
      InputImageSpacingType   spacing   = source->GetSpacing();
      InputImageDirectionType direction = source->GetDirection();
      
      target->SetRegions(region);
      target->SetOrigin(origin);
      target->SetSpacing(spacing);
      target->SetDirection(direction);
      target->Allocate();      
    }
  
  ImageRegionConstIterator<InputImageType> sourceIterator(source, source->GetLargestPossibleRegion());
  ImageRegionIterator<InputImageType>      targetIterator(target, target->GetLargestPossibleRegion());
  
  for (sourceIterator.GoToBegin(), targetIterator.GoToBegin();
       !sourceIterator.IsAtEnd() && !targetIterator.IsAtEnd();
       ++sourceIterator, ++targetIterator)
    {
      targetIterator.Set(sourceIterator.Get());
    }   
}

template < typename TInputImageType >
typename MaskedImageRegistrationMethod<TInputImageType>::InputImageType*
MaskedImageRegistrationMethod<TInputImageType>
::GetFixedImageCopy()
{
  return m_FixedImageCopy.GetPointer();  
}

template < typename TInputImageType >
typename MaskedImageRegistrationMethod<TInputImageType>::InputImageType*
MaskedImageRegistrationMethod<TInputImageType>
::GetMovingImageCopy()
{
  return m_MovingImageCopy.GetPointer();
}

/*
 * The bit that does the wiring together.
 */
template < typename TInputImageType >
void
MaskedImageRegistrationMethod<TInputImageType>
::Initialize() throw (ExceptionObject)
{
  niftkitkDebugMacro(<<"Initialize():Started");

  InputImageConstPointer inputFixedImage = this->GetFixedImage();
  InputImageConstPointer inputMovingImage = this->GetMovingImage();
  InputImageConstPointer inputFixedMask = this->GetFixedMask();
  InputImageConstPointer inputMovingMask = this->GetMovingMask();

  InputImageConstPointer postRescaledFixedImage;
  postRescaledFixedImage = inputFixedImage;
  
  if (m_RescaleFixedImage)
    {    
      niftkitkDebugMacro(<<"Initialize():Rescaling fixed image to: " << m_RescaleFixedMinimum \
          << ", and: " << m_RescaleFixedMaximum \
          << ", using boundary value: " << m_RescaleFixedBoundaryValue \
          << ", between: " << m_RescaleFixedLowerThreshold \
          << ", and: " << m_RescaleFixedUpperThreshold \
          );

      m_FixedRescaler->SetOutputMinimum(m_RescaleFixedMinimum);
      m_FixedRescaler->SetOutputMaximum(m_RescaleFixedMaximum);
      m_FixedRescaler->SetInputLowerThreshold(m_RescaleFixedLowerThreshold);
      m_FixedRescaler->SetInputUpperThreshold(m_RescaleFixedUpperThreshold);
      m_FixedRescaler->SetOutputBoundaryValue(m_RescaleFixedBoundaryValue);
      m_FixedRescaler->SetInput(inputFixedImage);
      m_FixedRescaler->UpdateLargestPossibleRegion();
      postRescaledFixedImage = m_FixedRescaler->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Rescaling fixed image....DONE");
    }

  InputImageConstPointer postRescaledMovingImage;
  postRescaledMovingImage = inputMovingImage;
                        
  if (m_RescaleMovingImage)
    {    
      niftkitkDebugMacro(<<"Initialize():Rescaling moving image to: " << m_RescaleMovingMinimum \
          << ", and: " << m_RescaleMovingMaximum \
          << ", using boundary value: " << m_RescaleMovingBoundaryValue \
          << ", between: " << m_RescaleMovingLowerThreshold \
          << ", and: " << m_RescaleMovingUpperThreshold \
          );

      m_MovingRescaler->SetOutputMinimum(m_RescaleMovingMinimum);
      m_MovingRescaler->SetOutputMaximum(m_RescaleMovingMaximum);
      m_MovingRescaler->SetInputLowerThreshold(m_RescaleMovingLowerThreshold);
      m_MovingRescaler->SetInputUpperThreshold(m_RescaleMovingUpperThreshold);
      m_MovingRescaler->SetOutputBoundaryValue(m_RescaleMovingBoundaryValue);
      m_MovingRescaler->SetInput(inputMovingImage);
      m_MovingRescaler->UpdateLargestPossibleRegion();
      postRescaledMovingImage = m_MovingRescaler->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Rescaling moving image....DONE");
    }

  InputImageConstPointer postSmoothedFixedImage;
  postSmoothedFixedImage = postRescaledFixedImage;
  
  if (m_Sigma > 0)
    {
      niftkitkDebugMacro(<<"Initialize():Smoothing fixed image, Sigma=" << m_Sigma);
      m_FixedSmoother->SetInput(postRescaledFixedImage);
      m_FixedSmoother->SetSigma(m_Sigma);
      m_FixedSmoother->UpdateLargestPossibleRegion();
      postSmoothedFixedImage = m_FixedSmoother->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Smoothing fixed image....DONE");
    }
  
  InputImageConstPointer postSmoothedMovingImage;
  postSmoothedMovingImage = postRescaledMovingImage;
  
  if (m_Sigma > 0)
    {
      niftkitkDebugMacro(<<"Initialize():Smoothing moving image, Sigma=" << m_Sigma);
      m_MovingSmoother->SetInput(postRescaledMovingImage);
      m_MovingSmoother->SetSigma(m_Sigma);
      m_MovingSmoother->UpdateLargestPossibleRegion();
      postSmoothedMovingImage = m_MovingSmoother->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Smoothing moving image....DONE");
    }

  niftkitkDebugMacro(<<"Initialize():Starting to copy (potentially) rescaled, (potentially) smoothed, fixed image");
  this->CopyImage(postSmoothedFixedImage.GetPointer(), m_FixedImageCopy.GetPointer());
  niftkitkDebugMacro(<<"Initialize():Finished copying (potentially) rescaled, (potentially) smoothed, fixed image");

  niftkitkDebugMacro(<<"Initialize():Starting to copy (potentially) rescaled, (potentially) smoothed, moving image");
  this->CopyImage(postSmoothedMovingImage.GetPointer(), m_MovingImageCopy.GetPointer());
  niftkitkDebugMacro(<<"Initialize():Finished copying (potentially) rescaled, (potentially) smoothed, moving image");

  InputImageConstPointer postThresholdingFixedMask;
  postThresholdingFixedMask = inputFixedMask;
    
  if (m_ThresholdFixedMask && !m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Thresholding fixed mask to: " << m_FixedMaskMinimum << ", and: " << m_FixedMaskMaximum);
      m_FixedMaskThresholder->SetLowerThreshold(m_FixedMaskMinimum);
      m_FixedMaskThresholder->SetUpperThreshold(m_FixedMaskMaximum);
      m_FixedMaskThresholder->SetInsideValue(1);
      m_FixedMaskThresholder->SetOutsideValue(0);
      m_FixedMaskThresholder->SetInput(inputFixedMask);
      m_FixedMaskThresholder->UpdateLargestPossibleRegion();
      postThresholdingFixedMask = m_FixedMaskThresholder->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Thresholding fixed mask....DONE");
    }

  InputImageConstPointer postThresholdingMovingMask;
  postThresholdingMovingMask = inputMovingMask;
    
  if (m_ThresholdMovingMask && !m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Thresholding moving mask to: " << m_MovingMaskMinimum << ", and: " << m_MovingMaskMaximum);
      m_MovingMaskThresholder->SetLowerThreshold(m_MovingMaskMinimum);
      m_MovingMaskThresholder->SetUpperThreshold(m_MovingMaskMaximum);
      m_MovingMaskThresholder->SetInsideValue(1);
      m_MovingMaskThresholder->SetOutsideValue(0);
      m_MovingMaskThresholder->SetInput(inputMovingMask);
      m_MovingMaskThresholder->UpdateLargestPossibleRegion();
      postThresholdingMovingMask = m_MovingMaskThresholder->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Thresholding moving mask....DONE");
    }

  StructuringType element;
  element.SetRadius(1);
  element.CreateStructuringElement();

  InputImageConstPointer postDilationFixedMask;
  postDilationFixedMask = postThresholdingFixedMask;
  
  niftkitkDebugMacro(<<"Initialize():m_NumberOfDilations=" << m_NumberOfDilations );
  
  if (m_NumberOfDilations > 0 && !m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Dilating fixed mask: " <<  m_NumberOfDilations << " times");
      m_FixedMaskDilater->SetDilateValue(1);
      m_FixedMaskDilater->SetBackgroundValue(0);
      m_FixedMaskDilater->SetBoundaryToForeground(false);
      m_FixedMaskDilater->SetKernel(element);
      m_FixedMaskDilater->SetInput(postThresholdingFixedMask);
      
      for (unsigned int i = 0; i < m_NumberOfDilations; i++)
        {
          m_FixedMaskDilater->UpdateLargestPossibleRegion();
          if (i != m_NumberOfDilations -1)
            {
              typename InputImageType::Pointer image = m_FixedMaskDilater->GetOutput();
              image->DisconnectPipeline();
              m_FixedMaskDilater->SetInput(image);
            }
          niftkitkDebugMacro(<<"Initialize():Dilating fixed mask, done: " << i);
        }
      postDilationFixedMask = m_FixedMaskDilater->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Dilating fixed mask....DONE");
    }    

  InputImageConstPointer postDilationMovingMask;
  postDilationMovingMask = postThresholdingMovingMask;
  
  if (m_NumberOfDilations > 0 && !m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Dilating moving mask: " <<  m_NumberOfDilations << " times");
      m_MovingMaskDilater->SetDilateValue(1);
      m_MovingMaskDilater->SetBackgroundValue(0);
      m_MovingMaskDilater->SetBoundaryToForeground(false);
      m_MovingMaskDilater->SetKernel(element);
      m_MovingMaskDilater->SetInput(postThresholdingMovingMask);
      
      for (unsigned int i = 0; i < m_NumberOfDilations; i++)
        {
          m_MovingMaskDilater->UpdateLargestPossibleRegion();
          if (i != m_NumberOfDilations -1)
            {
              typename InputImageType::Pointer image = m_MovingMaskDilater->GetOutput();
              image->DisconnectPipeline();
              m_MovingMaskDilater->SetInput(image);
            }
          niftkitkDebugMacro(<<"Initialize():Dilating moving mask, done: " << i);
        }
      postDilationMovingMask = m_MovingMaskDilater->GetOutput();
      niftkitkDebugMacro(<<"Initialize():Dilating moving mask....DONE");
    }    

  InputImageConstPointer postMaskingFixedImage;
  postMaskingFixedImage = postSmoothedFixedImage;
  
  if (m_UseFixedMask && m_MaskImageDirectly && !m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Multiplying fixed mask by fixed image");
      m_FixedImageMuliplier->SetInput1(postSmoothedFixedImage);
      m_FixedImageMuliplier->SetInput2(postDilationFixedMask);
      m_FixedImageMuliplier->UpdateLargestPossibleRegion();
      postMaskingFixedImage = m_FixedImageMuliplier->GetOutput();
      m_FixedImageMuliplier->Update();
      niftkitkDebugMacro(<<"Initialize():Multiplying fixed mask by fixed image....DONE");
    }

  InputImageConstPointer postMaskingMovingImage;
  postMaskingMovingImage = postSmoothedMovingImage;
  
  if (m_UseMovingMask && m_MaskImageDirectly && !m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Multiplying moving mask by moving image");
      m_MovingImageMultiplier->SetInput1(postSmoothedMovingImage);
      m_MovingImageMultiplier->SetInput2(postDilationMovingMask);
      m_MovingImageMultiplier->UpdateLargestPossibleRegion();
      postMaskingMovingImage = m_MovingImageMultiplier->GetOutput();
      m_MovingImageMultiplier->Update();
      niftkitkDebugMacro(<<"Initialize():Multiplying moving mask by moving image....DONE");
    }

  niftkitkDebugMacro(<<"Initialize():Fixed image, size: " << postMaskingFixedImage->GetLargestPossibleRegion().GetSize() \
    << ", spacing: " << postMaskingFixedImage->GetSpacing()  \
    << ", origin: " << postMaskingFixedImage->GetOrigin() \
    << ", direction:\n" << postMaskingFixedImage->GetDirection());

  niftkitkDebugMacro(<<"Initialize():Moving image, size: " << postMaskingMovingImage->GetLargestPossibleRegion().GetSize() \
    << ", spacing: " << postMaskingMovingImage->GetSpacing()  \
    << ", origin: " << postMaskingMovingImage->GetOrigin() \
    << ", direction:\n" << postMaskingMovingImage->GetDirection());
     
  this->SetFixedImage(postMaskingFixedImage);
  this->SetMovingImage(postMaskingMovingImage);
  
  if (m_UseFixedMask && !m_MaskImageDirectly && !m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Setting dilated mask (" << m_NumberOfDilations \
          << " dilations) as fixed mask on similarity measure");

      m_FixedMaskCaster->SetInput(postDilationFixedMask);
      m_FixedMaskCaster->UpdateLargestPossibleRegion();
      m_FixedMasker->SetImage(m_FixedMaskCaster->GetOutput());
      this->GetMetric()->SetFixedImageMask(m_FixedMasker);      
    }
  else
    {
      niftkitkDebugMacro(<<"Initialize():Not using fixed mask on similarity measure");
    }
  
  if (m_UseMovingMask && !m_MaskImageDirectly && !m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Setting dilated mask (" << m_NumberOfDilations \
          << " dilations) as moving mask on similarity measure");

      m_MovingMaskCaster->SetInput(postDilationMovingMask);
      m_MovingMaskCaster->UpdateLargestPossibleRegion();
      m_MovingMasker->SetImage(m_MovingMaskCaster->GetOutput());
      this->GetMetric()->SetMovingImageMask(m_MovingMasker);
    }
  else
    { 
      niftkitkDebugMacro(<<"Initialize():Not using moving mask on similarity measure");
    }
    
  Superclass::Initialize();
  
  niftkitkDebugMacro(<<"Initialize():Finished.");
}


template < typename TInputImageType >
void 
MaskedImageRegistrationMethod<TInputImageType>
::SetFixedMask( const InputImageType * fixedMask )
{
  niftkitkDebugMacro(<<"Setting Fixed Mask to " << fixedMask);

  if (this->m_FixedMask.GetPointer() != fixedMask ) 
    { 
      this->m_FixedMask = fixedMask;
      this->Modified(); 
    } 
}

template < typename TInputImageType >
void 
MaskedImageRegistrationMethod<TInputImageType>
::SetMovingMask( const InputImageType * movingMask )
{
  niftkitkDebugMacro(<<"Setting Moving Mask to " << movingMask);

  if (this->m_MovingMask.GetPointer() != movingMask ) 
    { 
      this->m_MovingMask = movingMask;
      this->Modified(); 
    } 
}

} // end namespace itk


#endif
