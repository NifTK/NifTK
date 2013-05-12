/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKLocalSimilarityMeasureGradientDescentOptimizer_TXX_
#define ITKLocalSimilarityMeasureGradientDescentOptimizer_TXX_

#include "itkLocalSimilarityMeasureGradientDescentOptimizer.h"
#include <itkResampleImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkAddImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkIdentityTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <ConversionUtils.h>
#include <itkAbsImageFilter.h>

#include <itkLogHelper.h>
#include <itkUCLMacro.h>

namespace itk
{
template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
LocalSimilarityMeasureGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::LocalSimilarityMeasureGradientDescentOptimizer()
{
  this->m_Maximize = true; // normally we will be doing normalized mutual info.
  this->m_Value = 0;  
  this->m_Stop = false;
  this->m_MaximumNumberOfIterations = 100;
  this->m_CurrentIteration = 0;
  this->m_StepSize = 5.0;
  this->m_MinimumStepSize = 0.001;
  this->m_IteratingStepSizeReductionFactor = 0.5;
  this->m_RegriddingStepSizeReductionFactor = 0.5;
  this->m_JacobianBelowZeroStepSizeReductionFactor = 0.5;
  this->m_MinimumDeformationMagnitudeThreshold = 0.05;
  this->m_MinimumJacobianThreshold = 0.5;
  this->m_MinimumSimilarityChangeThreshold = 0.0005; 
  this->m_CheckSimilarityMeasure = true; 
  this->m_WriteNextParameters = false;
  this->m_NextParametersFileName = "tmp.next.params";
  this->m_NextParametersFileExt = "vtk";
  this->m_WriteDeformationField = false;
  this->m_DeformationFieldFileName = "tmp.next.field";
  this->m_DeformationFieldFileExt = "vtk";
  this->m_CheckMinDeformationMagnitudeThreshold = true;  
  this->m_RegriddingNumber = 0;
  this->m_WriteRegriddedImage = false;
  this->m_RegriddedImageFileName = "tmp.regridded";
  this->m_RegriddedImageFileExt = "nii";
  this->m_CheckJacobianBelowZero = true;
  this->m_RegriddedMovingImagePadValue = 0;
  this->m_IsAbsRegriddedImage = false; 
  this->m_IsPropagateRegriddedMovingImage = false; 
  
  m_RegriddingResampler = ResampleFilterType::New();
  
  niftkitkDebugMacro(<< "LocalSimilarityMeasureGradientDescentOptimizer():Constructed, Maximize=" << m_Maximize \
    << ", Value=" << m_Value \
    << ", Stop=" << m_Stop \
    << ", MaxIters=" << m_MaximumNumberOfIterations \
    << ", CurrIter=" << m_CurrentIteration \
    << ", Step=" << m_StepSize \
    << ", MinStep=" << m_MinimumStepSize \
    << ", IterReductFactor=" << m_IteratingStepSizeReductionFactor \
    << ", RegridReductFactor=" << m_RegriddingStepSizeReductionFactor \
    << ", JacobianReductFactor=" << m_JacobianBelowZeroStepSizeReductionFactor \
    << ", DefMagThresh=" << m_MinimumDeformationMagnitudeThreshold \
    << ", MinJacThresh=" << m_MinimumJacobianThreshold \
    << ", MinimumSimilarityChangeThreshold=" << m_MinimumSimilarityChangeThreshold \
    << ", WriteNextParameters=" << m_WriteNextParameters \
    << ", NextParametersFileName=" << m_NextParametersFileName \
    << ", NextParametersFileExt=" << m_NextParametersFileExt \
    << ", m_WriteDeformationField=" << m_WriteDeformationField \
    << ", m_DeformationFieldFileName=" << m_DeformationFieldFileName \
    << ", m_DeformationFieldFileExt=" << m_DeformationFieldFileExt \
    << ", m_CheckMinDeformationMagnitudeThreshold=" << m_CheckMinDeformationMagnitudeThreshold \
    << ", m_RegriddingNumber=" << m_RegriddingNumber \
    << ", m_WriteRegriddedImage=" << m_WriteRegriddedImage \
    << ", m_RegriddedImageFileName=" << m_RegriddedImageFileName \
    << ", m_RegriddedImageFileExt=" << m_RegriddedImageFileExt \
    << ", m_CheckJacobianBelowZero=" << m_CheckJacobianBelowZero \
    << ", m_RegriddedMovingImagePadValue=" << m_RegriddedMovingImagePadValue \
    );
}

/*
 * PrintSelf
 */
template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Maximize=" << m_Maximize << std::endl;
  os << indent << "Value=" << m_Value << std::endl;
  os << indent << "Stop=" << m_Stop << std::endl;
  os << indent << "MaxIters=" << m_MaximumNumberOfIterations << std::endl;
  os << indent << "CurrIter=" << m_CurrentIteration << std::endl;
  os << indent << "Step=" << m_StepSize << std::endl;
  os << indent << "MinStep=" << m_MinimumStepSize << std::endl;
  os << indent << "IterReductFactor=" << m_IteratingStepSizeReductionFactor << std::endl;
  os << indent << "RegridReductFactor=" << m_RegriddingStepSizeReductionFactor << std::endl;
  os << indent << "JacobianReductFactor=" << m_JacobianBelowZeroStepSizeReductionFactor  << std::endl;
  os << indent << "DefMagThresh=" << m_MinimumDeformationMagnitudeThreshold << std::endl;
  os << indent << "MinJacThresh=" << m_MinimumJacobianThreshold << std::endl;
  os << indent << "MinimumSimilarityChangeThreshold=" << m_MinimumSimilarityChangeThreshold << std::endl;
  os << indent << "WriteNextParameters=" << m_WriteNextParameters << std::endl;
  os << indent << "NextParametersFileName=" << m_NextParametersFileName << std::endl;
  os << indent << "NextParametersFileExt=" << m_NextParametersFileName << std::endl;
  os << indent << "WriteDeformationField=" << m_WriteDeformationField << std::endl;
  os << indent << "DeformationFieldFileName=" << m_DeformationFieldFileName << std::endl;
  os << indent << "DeformationFieldFileExt=" << m_DeformationFieldFileExt << std::endl;
  os << indent << "CheckMinDeformationMagnitudeThreshold=" << m_CheckMinDeformationMagnitudeThreshold << std::endl;  
  os << indent << "RegriddingNumber=" << m_RegriddingNumber << std::endl;
  os << indent << "WriteRegriddedImage=" << m_WriteRegriddedImage << std::endl;
  os << indent << "RegriddedImageFileName=" << m_RegriddedImageFileName << std::endl;
  os << indent << "RegriddedImageFileExt=" << m_RegriddedImageFileExt << std::endl;
  os << indent << "CheckJacobianBelowZero=" << m_CheckJacobianBelowZero << std::endl;
  os << indent << "RegriddedMovingImagePadValue=" << m_RegriddedMovingImagePadValue << std::endl;
  
}

template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::StopOptimization( void )
{
  niftkitkDebugMacro(<< "StopOptimization()");
  m_Stop = true;
  InvokeEvent(EndEvent());
}

/**
 * Start the optimization
 */
template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::StartOptimization( void )
{
  niftkitkDebugMacro(<< "StartOptimization():Started");

  if (m_DeformableTransform.IsNull())
    {
      itkExceptionMacro(<< "The deformable transform is null, please inject one.");
    }
  
  m_ImageToImageMetric = dynamic_cast<ImageToImageMetricPointer>(this->GetCostFunction());
  if (m_ImageToImageMetric == 0)
    {
      itkExceptionMacro(<< "Cannot cast image to image metric.");
    }
  
  m_FixedImage = const_cast<FixedImagePointer>(m_ImageToImageMetric->GetFixedImage());
  if (m_FixedImage == 0)
    {
      itkExceptionMacro(<< "Cannot cast fixed image");
    }
  
  m_MovingImage = const_cast<MovingImagePointer>(m_ImageToImageMetric->GetMovingImage());
  if (m_MovingImage == 0)
    {
      itkExceptionMacro(<< "Cannot cast moving image");
    }

  this->m_CurrentIteration = 0;
  this->SetCurrentPosition( this->GetInitialPosition() );  
  this->ResumeOptimization();
  niftkitkDebugMacro(<< "StartOptimization():Finished");
}


template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::ResumeOptimization( void )
{
  unsigned long int numberOfParameters = this->m_DeformableTransform->GetNumberOfParameters();
  
  niftkitkInfoMacro(<< "ResumeOptimization():Started, optimising:" << numberOfParameters << " parameters");
  InvokeEvent( StartEvent() );
  
  double stepSize = m_StepSize;
  double minJacobian = 0;
  
  this->m_NextParameters.SetSize(numberOfParameters);  
  this->m_CurrentIteration = 0;

  // ReGrid first of all.
  this->ReGrid(false);
  
  niftkitkInfoMacro(<< "ResumeOptimization():Starting with initial value of:" << niftk::ConvertToString(this->m_Value));
  
  // Stop can be set by another thread WHILE we are iterating (potentially).
  m_Stop = false;
  
  // Subclasses can initialize at this point
  this->Initialize();
  
  double nextValue = std::numeric_limits<double>::max(); 
  
  while( !m_Stop ) 
  {
    m_CurrentIteration++;
    
    double maxDeformationChange = 0.0;
  
    if (m_CurrentIteration > m_MaximumNumberOfIterations)
      {
        niftkitkInfoMacro(<< "ResumeOptimization():Hit max iterations:" << m_MaximumNumberOfIterations);
        StopOptimization();
        break;
      }
    
    if (m_StepSize < m_MinimumStepSize)
      {
        niftkitkInfoMacro(<< "ResumeOptimization():Gone below min step size:" << m_MinimumStepSize);
        StopOptimization();
        break;
      }

    // Reset the next position array.
    this->m_NextParameters.Fill(0);
  
    // Calculate the next set of parameters, which may or may not be 'accepted'.
    nextValue = CalculateNextStep(m_CurrentIteration, this->m_Value, this->GetCurrentPosition(), this->m_NextParameters);
    
    // Set them onto the transform. We dont need this, as computing the similarity measure sets the transform.
    // this->m_DeformableTransform->SetParameters(this->m_NextParameters);

    // So far so good, so compute new similarity measure.                                                                               
    if (nextValue == std::numeric_limits<double>::max())
    {
      nextValue = this->GetCostFunction()->GetValue(this->m_NextParameters);
    }
    else
    {
      // Need this for checking Jacobian. 
      this->m_DeformableTransform->SetParameters(this->m_NextParameters);
    }

    // Write the parameters to file.
    if (m_WriteNextParameters)
      {
        this->m_DeformableTransform->WriteParameters(m_NextParametersFileName + "." + niftk::ConvertToString((int)m_CurrentIteration) + "." + m_NextParametersFileExt);
      }

    // Write this deformation field to file. 
    // For Fluid, this is the same as the dump of the parameters above.
    // For FFD, the one above will be the control point grid, so this is the actual deformation field.
    if (m_WriteDeformationField)
      {
        this->m_DeformableTransform->WriteTransform(m_DeformationFieldFileName + "." + niftk::ConvertToString((int)m_CurrentIteration) + "." + m_DeformationFieldFileExt);    
      }
    
    // Calculate the max deformation change. 
    for (unsigned long int i = 0; i < numberOfParameters; i++)
      {
        double change = fabs(this->m_NextParameters.GetElement(i)-this->m_CurrentPosition.GetElement(i));
        
        if (change > maxDeformationChange)
          {
            maxDeformationChange = change;  
          }
           
      }
    niftkitkDebugMacro(<< "ResumeOptimization(): maxDeformationChange=" << maxDeformationChange << " and m_MinimumDeformationMagnitudeThreshold=" << this->m_MinimumDeformationMagnitudeThreshold << ", and m_CheckMinDeformationMagnitudeThreshold=" << m_CheckMinDeformationMagnitudeThreshold);
    
    minJacobian = this->m_DeformableTransform->ComputeMinJacobian();

    if (m_CheckJacobianBelowZero && minJacobian < 0.0)
      {
        this->m_StepSize *= this->m_JacobianBelowZeroStepSizeReductionFactor;
        
        niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << m_CurrentIteration \
          << ", Negative Jacobian " << minJacobian << ", so rejecting this change, reducing step size by:" \
          << m_JacobianBelowZeroStepSizeReductionFactor \
          << ", to:" << this->m_StepSize \
          << ", and continuing");
      }
    else if(m_CheckMinDeformationMagnitudeThreshold && maxDeformationChange < this->m_MinimumDeformationMagnitudeThreshold)
      {
        niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << m_CurrentIteration \
          << ", maxDeformation was:" << maxDeformationChange \
          << ", threshold was:" << this->m_MinimumDeformationMagnitudeThreshold \
          << ", so I'm rejecting this change, reducing step size and continuing");

        this->m_StepSize *= m_IteratingStepSizeReductionFactor;
      }
    else
      {
        niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << m_CurrentIteration \
          << ", currentValue=" << niftk::ConvertToString(this->m_Value) \
          << ", nextValue=" << niftk::ConvertToString(nextValue) \
          << ", minJacobian=" <<  niftk::ConvertToString(minJacobian) \
          << ", maxDeformationChange=" <<  niftk::ConvertToString(maxDeformationChange));
      
        if (  (this->m_Maximize && nextValue > this->m_Value)
           || (!this->m_Maximize && nextValue < this->m_Value) 
           || !this->m_CheckSimilarityMeasure)
          {

            niftkitkInfoMacro(<< "ResumeOptimization():Maximize:" << this->m_Maximize \
              << ", currentValue:" << niftk::ConvertToString(this->m_Value) \
              << ", nextValue:" << niftk::ConvertToString(nextValue) \
              << ", so its better!");
          
            if (fabs(nextValue - this->m_Value) <  this->m_MinimumSimilarityChangeThreshold
                && this->m_CheckSimilarityMeasure)
              {
                niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << m_CurrentIteration \
                  << ", similarity change was:" << fabs(nextValue - this->m_Value) \
                  << ", threshold was:" << this->m_MinimumSimilarityChangeThreshold \
                  << ", so I'm accepting this change, but then stopping.");
                this->SetCurrentPosition(this->m_NextParameters);
                this->m_Value = nextValue;
                StopOptimization();
                break;
              }
            
            bool isAcceptNextParameterAccordingJacobian = true; 
            // Check if we need to regrid only after we got a better similarity value. 
            if (minJacobian < this->m_MinimumJacobianThreshold)
              {
                if (this->m_DeformableTransform->IsRegridable())
                  {
                    niftkitkInfoMacro(<< "ResumeOptimization():Jacobian is lower than threshold:" <<  m_MinimumJacobianThreshold \
                      << ", so accepting this change, reducing step size by:" \
                      << m_RegriddingStepSizeReductionFactor \
                      << ", to:" << this->m_StepSize \
                      << ", regridding and continuing");

                    this->m_StepSize *= m_RegriddingStepSizeReductionFactor;

                    this->ReGrid(true);
                    isAcceptNextParameterAccordingJacobian = false; 
                  }
                else
                  {
                    niftkitkInfoMacro(<< "ResumeOptimization():Jacobian is lower than threshold:" <<  m_MinimumJacobianThreshold \
                      << ", but this->m_DeformableTransform->IsRegridable() is false, so I'm not going to regrid.");
                  }
              } 
            
            if (isAcceptNextParameterAccordingJacobian)
            {
              this->SetCurrentPosition(this->m_NextParameters);
              this->m_Value = nextValue;
            }
            
          }
        else
          {
            // Revert the transformed moving image to the current parameter. 
            // We already have the current value, no need to re-evaluate it.
            // this->GetCostFunction()->GetValue(this->GetCurrentPosition());
            
            this->m_StepSize *= m_IteratingStepSizeReductionFactor;

            niftkitkInfoMacro(<< "ResumeOptimization():Maximize:" << this->m_Maximize \
              << ", currentValue:" << niftk::ConvertToString(this->m_Value) \
              << ", nextValue:" << niftk::ConvertToString(nextValue) \
              << ", so its no better, so rejecting this change, reducing step size by:" \
              << m_IteratingStepSizeReductionFactor \
              << ", to:" << this->m_StepSize \
              << " and continuing");
          }
      } 

  } // End main while loop.

  // Subclasses can cleanup at this point
  this->CleanUp();
  
  niftkitkInfoMacro(<< "ResumeOptimization():Aggregating transformation");
  
  // Add deformation field to existing regridding one.  
  if (this->m_DeformableTransform->IsRegridable())
    {
      ComposeJacobian(); 
      
      if (this->m_IsPropagateRegriddedMovingImage)
      {
        niftkitkInfoMacro(<< "ResumeOptimization(): propagate regridded moving image.");
        this->m_RegriddingResampler->SetTransform(this->m_DeformableTransform);
        this->m_RegriddingResampler->SetInterpolator(this->m_RegriddingInterpolator);      
        this->m_RegriddingResampler->SetInput(this->m_RegriddedMovingImage);
        this->m_RegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
        this->m_RegriddingResampler->SetDefaultPixelValue(m_RegriddedMovingImagePadValue);
        this->m_RegriddingResampler->Modified();  
        this->m_RegriddingResampler->UpdateLargestPossibleRegion();
        this->m_RegriddedMovingImage = m_RegriddingResampler->GetOutput();
        this->m_RegriddedMovingImage->DisconnectPipeline();
      }
      
      this->m_DeformableTransform->UpdateRegriddedDeformationParameters(this->m_RegriddedParameters, this->m_CurrentPosition); 
      this->m_CurrentPosition = this->m_RegriddedParameters; 
    }

  // Im resetting the m_Step size, so that if you repeatedly call 
  // the optimizer, it does actually do some optimizing.
  m_StepSize = stepSize;
  
  // Release memory. 
  this->m_NextParameters.SetSize(1);  
  this->m_RegriddedParameters.SetSize(1); 
  
  InvokeEvent( EndEvent() );
  niftkitkInfoMacro(<< "ResumeOptimization():Finished");
}


template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::ReGrid(bool isResetCurrentPosition)
{
  niftkitkDebugMacro(<< "ReGrid():Started");
  typedef ImageDuplicator<JacobianImageType> JacobianDuplicatorType; 
  
  unsigned long int numberOfParameters = this->m_CurrentPosition.GetSize();
  
  if (this->m_CurrentIteration == 0)
    {
      this->m_RegriddedParameters.SetSize(numberOfParameters);
      if (this->m_DeformableTransform->GetGlobalTransform() == NULL)
      {
        this->m_RegriddedParameters.Fill(0);
        niftkitkDebugMacro(<< "ReGrid():First iteration, so resizing regrid params to:" <<  this->m_RegriddedParameters.GetSize() \
            << ", and they are all zero.");
      }
      else
      {
         niftkitkDebugMacro(<< "ReGrid():First iteration, setting global transform.");
         this->m_DeformableTransform->InitialiseGlobalTransformation();
         for (unsigned int i = 0; i < numberOfParameters; i++)
           this->m_RegriddedParameters[i] = this->m_DeformableTransform->GetParameters()[i];
       }
    }
  
  if (isResetCurrentPosition)
  {
    niftkitkDebugMacro(<< "ReGrid():Adding current array to regrid array");
    
//    niftkitkDebugMacro(<< "ReGrid():m_RegriddedParameters before=" << m_RegriddedParameters);
//    niftkitkDebugMacro(<< "ReGrid():m_CurrentPosition=" << m_CurrentPosition);
    
    this->m_DeformableTransform->UpdateRegriddedDeformationParameters(this->m_RegriddedParameters, this->m_CurrentPosition); 

//    niftkitkDebugMacro(<< "ReGrid():m_RegriddedParameters after=" << m_RegriddedParameters);
  }

  // Set transform.
  if (this->m_CurrentIteration == 0)
  {
    this->m_DeformableTransform->SetParameters(this->m_RegriddedParameters);
    
    this->m_DeformableTransform->ComputeMinJacobian(); 
    typename JacobianDuplicatorType::Pointer duplicator = JacobianDuplicatorType::New(); 
    duplicator->SetInputImage(this->m_DeformableTransform->GetJacobianImage()); 
    duplicator->Update(); 
    this->m_ComposedJacobian = duplicator->GetOutput(); 
    this->m_ComposedJacobian->DisconnectPipeline(); 
  }
  else
  {
    if (!this->m_IsPropagateRegriddedMovingImage)
    {
      this->m_DeformableTransform->SetParameters(this->m_RegriddedParameters);
    }
    else
    {
      niftkitkDebugMacro(<< "ReGrid():setting current position for propagating moving image.");
      this->m_DeformableTransform->SetParameters(this->m_CurrentPosition);
    }
    ComposeJacobian(); 
  }

//    #ifdef(DEBUG)
      niftkitkDebugMacro(<< "ReGrid():Check before resample, minJacobian=" \
        <<  this->m_DeformableTransform->ComputeMinJacobian() \
        << ", maxJacobian=" << this->m_DeformableTransform->ComputeMaxJacobian() \
        << ", minDeformation=" << this->m_DeformableTransform->ComputeMinDeformation() \
        << ", maxDeformation=" << this->m_DeformableTransform->ComputeMaxDeformation() );
  //  #endif

  // Transform the original moving image. So, if isResetCurrentPosition we assume
  // that this->m_DeformableTransform has an up to date transformation.
  if (this->m_DeformableTransform->IsIdentity())
    {
      niftkitkDebugMacro(<< "ReGrid():Regridding using identity transform, and nearest neighbour interpolator, as its quicker than using m_DeformableTransform");
      typedef IdentityTransform<double, Dimension> IdentityTransformType;
      typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();
      
      typedef NearestNeighborInterpolateImageFunction<TFixedImage, double> NearestInterpolatorType;
      typename NearestInterpolatorType::Pointer nearestInterpolator = NearestInterpolatorType::New();
      
      this->m_RegriddingResampler->SetTransform(identityTransform);
      this->m_RegriddingResampler->SetInterpolator(nearestInterpolator);
    }
  else
    {
      this->m_RegriddingResampler->SetTransform(this->m_DeformableTransform);
      this->m_RegriddingResampler->SetInterpolator(this->m_RegriddingInterpolator);      
    }

  if (this->m_CurrentIteration == 0)
  {
    this->m_RegriddingResampler->SetInput(this->m_MovingImage);
  }
  else
  {
    if (!this->m_IsPropagateRegriddedMovingImage)
    {
      this->m_RegriddingResampler->SetInput(this->m_MovingImage);
    }
    else
    {
      this->m_RegriddingResampler->SetInput(this->m_RegriddedMovingImage);
    }
  }
  this->m_RegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
  this->m_RegriddingResampler->SetDefaultPixelValue(m_RegriddedMovingImagePadValue);
  this->m_RegriddingResampler->Modified();  
  this->m_RegriddingResampler->UpdateLargestPossibleRegion();
  
  this->m_RegriddedMovingImage = m_RegriddingResampler->GetOutput();
  this->m_RegriddedMovingImage->DisconnectPipeline();
  
  if (this->m_IsAbsRegriddedImage)
  {
    typedef AbsImageFilter<TFixedImage, TFixedImage> AbsImageFilterType; 
    typename AbsImageFilterType::Pointer absImageFilter = AbsImageFilterType::New(); 
    
    absImageFilter->SetInput(this->m_RegriddedMovingImage); 
    absImageFilter->Update(); 
    this->m_RegriddedMovingImage = absImageFilter->GetOutput(); 
    this->m_RegriddedMovingImage->DisconnectPipeline();
  }

  niftkitkDebugMacro(<< "ReGrid():m_RegriddedMovingImage is currently at address=" << &this->m_RegriddedMovingImage \
    << ", size=" << this->m_RegriddedMovingImage->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << this->m_RegriddedMovingImage->GetSpacing()
    << ", origin=" << this->m_RegriddedMovingImage->GetOrigin() \
    << ", direction=\n" << this->m_RegriddedMovingImage->GetDirection());

  niftkitkDebugMacro(<< "ReGrid():m_FixedImage is currently at address=" << this->m_FixedImage \
    << ", size=" << this->m_FixedImage->GetLargestPossibleRegion().GetSize() \
    << ", spacing=" << this->m_FixedImage->GetSpacing()
    << ", origin=" << this->m_FixedImage->GetOrigin() \
    << ", direction=\n" << this->m_FixedImage->GetDirection());

  // Reset the transform, set transform to Identity, 
  // as its quicker than reinterpolating a BSpline grid.
  if (isResetCurrentPosition)
  {
    this->m_CurrentPosition.Fill(0);
    this->m_DeformableTransform->SetIdentity();
  }


      niftkitkDebugMacro(<< "ReGrid():Check after resample, minJacobian=" \
        <<  this->m_DeformableTransform->ComputeMinJacobian() \
        << ", maxJacobian=" << this->m_DeformableTransform->ComputeMaxJacobian() \
        << ", minDeformation=" << this->m_DeformableTransform->ComputeMinDeformation() \
        << ", maxDeformation=" << this->m_DeformableTransform->ComputeMaxDeformation() );

      // Calculate min and max image values in fixed image.
      ImageRegionConstIterator<FixedImageType> fiIt(this->m_FixedImage,
                                                    this->m_FixedImage->GetLargestPossibleRegion());
      fiIt.GoToBegin();
      FixedImagePixelType minFixed = fiIt.Value();
      FixedImagePixelType maxFixed = fiIt.Value();
      ++fiIt;
      while ( !fiIt.IsAtEnd() )
        {
          FixedImagePixelType value = fiIt.Value();

          if (value < minFixed)
            {
              minFixed = value;
            }
              else if (value > maxFixed)
            {
              maxFixed = value;
            }
          ++fiIt;
        }
      
      // Calculate min and max image values in moving image.
      ImageRegionConstIterator<MovingImageType> miIt(this->m_RegriddedMovingImage,
                                                     this->m_RegriddedMovingImage->GetLargestPossibleRegion());
      miIt.GoToBegin();
      MovingImagePixelType minMoving = miIt.Value();
      MovingImagePixelType maxMoving = miIt.Value();
      ++miIt;
      while ( !miIt.IsAtEnd() )
        {
          MovingImagePixelType value = miIt.Value();

          if (value < minMoving)
            {
              minMoving = value;
            }
          else if (value > maxMoving)
            {
              maxMoving = value;
            }
          ++miIt;
        }
      niftkitkDebugMacro(<< std::string("ReGrid():Checking output of regridding:")
        + "fixedLower:" + niftk::ConvertToString(minFixed)
        + ",fixedUpper:" + niftk::ConvertToString(maxFixed)
        + ",movingLower:" + niftk::ConvertToString(minMoving)
        + ",movingUpper:" + niftk::ConvertToString(maxMoving));

  // Write regridded image to file for debug purposes.
  if (this->m_WriteRegriddedImage)
    {
      typedef ImageFileWriter<TFixedImage> RegriddedImageFileWriterType;
      typename RegriddedImageFileWriterType::Pointer regriddedWriter = RegriddedImageFileWriterType::New();

      regriddedWriter->SetInput(this->m_RegriddedMovingImage);
      
      std::string tmpFilename = this->m_RegriddedImageFileName + "." + niftk::ConvertToString((int)m_RegriddingNumber) \
        + "." + this->m_RegriddedImageFileExt;
      
      niftkitkDebugMacro(<< "ReGrid():Writing image to:" << tmpFilename);
      
      regriddedWriter->SetFileName(tmpFilename);
      regriddedWriter->Update();
      
      // Increment the image number.
      this->m_RegriddingNumber++;
    }

  // Recompute the similarity.
  (const_cast<ImageToImageMetricType*>(m_ImageToImageMetric))->SetMovingImage(this->m_RegriddedMovingImage);
  
  if (!isResetCurrentPosition)
  {
    this->m_DeformableTransform->SetParameters(this->m_CurrentPosition);
  }
  
  // Reintitialise the bins. 
  (const_cast<ImageToImageMetricType*>(m_ImageToImageMetric))->Initialize();
  
  this->m_Value = m_ImageToImageMetric->GetValue(this->m_DeformableTransform->GetParameters());
  
  niftkitkDebugMacro(<< "ReGrid():Setting current value to:" << this->m_Value);

      niftkitkDebugMacro(<< "ReGrid():Check after resetting to current position, minJacobian=" \
        <<  this->m_DeformableTransform->ComputeMinJacobian() \
        << ", maxJacobian=" << this->m_DeformableTransform->ComputeMaxJacobian() \
        << ", minDeformation=" << this->m_DeformableTransform->ComputeMinDeformation() \
        << ", maxDeformation=" << this->m_DeformableTransform->ComputeMaxDeformation() );

#if 0
  char filename[100];
  typedef itk::Image< unsigned char, TFixedImage::ImageDimension > OutputImageType;
  
  //typedef itk::RescaleIntensityImageFilter<TMovingImage, OutputImageType> RescalerType;
  typedef itk::RescaleIntensityImageFilter<TFixedImage, OutputImageType> RescalerType;
  typename RescalerType::Pointer intensityRescaler = RescalerType::New();
  
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  //typedef itk::ImageFileWriter< TFixedImage >  WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  
  //typedef itk::CastImageFilter< TMovingImage, OutputImageType > CastFilterType;
  typedef itk::CastImageFilter< OutputImageType, OutputImageType > CastFilterType;
  typename CastFilterType::Pointer castFilter = CastFilterType::New();

  intensityRescaler->SetInput(this->m_RegriddedMovingImage);
  intensityRescaler->SetOutputMinimum(0);
  intensityRescaler->SetOutputMaximum(255);
  
  castFilter->SetInput(intensityRescaler->GetOutput());
  
  writer->SetInput(castFilter->GetOutput());
  //writer->SetInput(this->m_RegriddedMovingImage);
  sprintf(filename, "regrid%03u.png", this->m_CurrentIteration);
  writer->SetFileName(filename);
  writer->Update();
#endif     

  niftkitkDebugMacro(<< "ReGrid():Finished.");

}




} // namespace itk.

#endif
