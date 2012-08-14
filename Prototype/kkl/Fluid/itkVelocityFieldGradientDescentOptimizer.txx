/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-27 13:54:26 +0100 (Fri, 27 May 2011) $
 Revision          : $Revision: 6300 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKVelocityFieldGradientDescentOptimizer_TXX_
#define ITKVelocityFieldGradientDescentOptimizer_TXX_

#include "itkVelocityFieldGradientDescentOptimizer.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageDuplicator.h"
#include <vector>

#include "itkLogHelper.h"

#define SHOOT

namespace itk
{
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::VelocityFieldGradientDescentOptimizer()
{
  this->m_StartingStepSize = 0.0;
  this->m_NormaliseStepSize = true; 
  this->m_MinimumDeformationMaximumIterations = -1; 
  this->m_CurrentMinimumDeformationIterations = 0; 
  this->m_BestIteration = 0; 
  this->m_WorseNumberOfIterationsAllowed = 20; 
  this->m_NormalisationFactor = 1.; 
  
  niftkitkDebugMacro(<< "VelocityFieldGradientDescentOptimizer():Constructed");
}

template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
void
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::StartOptimization( void )
{
  niftkitkDebugMacro(<< "StartVelocityFieldOptimization():");
  
  this->m_StartingStepSize = this->m_StepSize; 
  this->m_NormaliseStepSize = true; 
  this->m_CurrentMinimumDeformationIterations = 0; 
  this->m_CalculateVelocityFeild = true; 
  
  if (this->m_DeformableTransform.IsNull())
  {
    itkExceptionMacro(<< "The deformable transform is null, please inject one.");
  }
  
  this->m_ImageToImageMetric = dynamic_cast<typename Superclass::ImageToImageMetricPointer>(this->GetCostFunction());
  if (this->m_ImageToImageMetric == 0)
  {
    itkExceptionMacro(<< "Cannot cast image to image metric.");
  }
  
  this->m_FixedImage = const_cast<typename Superclass::FixedImagePointer>(this->m_ImageToImageMetric->GetFixedImage());
  if (this->m_FixedImage == 0)
  {
    itkExceptionMacro(<< "Cannot cast fixed image");
  }
  
  this->m_MovingImage = const_cast<typename Superclass::MovingImagePointer>(this->m_ImageToImageMetric->GetMovingImage());
  if (this->m_MovingImage == 0)
  {
    itkExceptionMacro(<< "Cannot cast moving image");
  }

  this->m_CurrentIteration = 0;
  this->m_BestIteration = 0; 
  
  typename Superclass::ScalesType scales(1);
  scales.Fill(1.0);
  this->SetScales(scales);
  
  if (GetInitialDeformableParameters().IsNull())
  {
    this->m_CurrentDeformableParameters = GetMovingImageVelocityFieldDeformableTransform()->GetDeformableParameters(); 
  }
  else
  {
    this->m_CurrentDeformableParameters = GetInitialDeformableParameters(); 
  }
  this->m_NextDeformableParameters = DeformableTransformType::DuplicateDeformableParameters(this->m_CurrentDeformableParameters); 
  this->m_RegriddedDeformableParameters = DeformableTransformType::DuplicateDeformableParameters(this->m_CurrentDeformableParameters); 
  
  this->m_StepSizeImage.clear(); 
  this->m_StepSizeNormalisationFactorImage.clear(); 
  this->m_PreviousVelocityFieldGradient.clear(); 
  
  // this->m_PreviousVelocityFieldGradient = DeformableTransformType::DuplicateDeformableParameters(this->m_CurrentDeformableParameters); 
  for (int i = 0; i < GetMovingImageVelocityFieldDeformableTransform()->GetNumberOfVelocityField(); i++)
  {
    typename StepSizeImageType::Pointer image = StepSizeImageType::New(); 
    typename StepSizeImageType::RegionType region = this->m_FixedImage->GetLargestPossibleRegion().GetSize(); 
    
    if (i == 0)
    {
#ifndef SHOOT        
      image->SetRegions(region); 
      image->Allocate(); 
      image->FillBuffer(this->m_StepSize); 
      this->m_StepSizeImage.push_back(image); 
      
      typename StepSizeImageType::Pointer normalisationImage = StepSizeImageType::New(); 
      normalisationImage->SetRegions(region); 
      normalisationImage->Allocate(); 
      normalisationImage->FillBuffer(0.); 
      this->m_StepSizeNormalisationFactorImage.push_back(normalisationImage); 
#endif     
    
      this->m_PreviousVelocityFieldGradient.push_back(DeformableTransformType::DuplicateDeformableParameters(this->m_CurrentDeformableParameters)); 
    }
    
#ifndef SHOOT        
    DeformableTransformType::SaveField(this->m_PreviousVelocityFieldGradient[0], VelocityFieldDeformableTransformFilename::GetPreviousVelocityFieldGradientFilename(i)); 
    SaveStepSizeImage(this->m_StepSizeImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeImageFilename(i)); 
    SaveStepSizeImage(this->m_StepSizeNormalisationFactorImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeNormalisationFactorFilename(i)); 
#endif    
  }
  
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetFixedImage(this->m_FixedImage);
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetMovingImage(this->m_MovingImage);
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetTransform(GetMovingImageVelocityFieldDeformableTransform()); 
  this->m_Value = this->m_ImageToImageMetric->GetValue(this->m_DeformableTransform->GetParameters()); 
  
  this->ResumeOptimization();
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
void
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  
  if (!m_ForceFilter.IsNull())
  {
    os << indent << "ForceFilter=" << m_ForceFilter << std::endl;
  }
  if (!m_FluidPDESolver.IsNull())
  {
    os << indent << "FluidPDESolver=" << m_FluidPDESolver << std::endl;
  }
}

#if 0
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
double
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::CalculateNextStep(int iterationNumber, double currentSimilarity, typename DeformableTransformType::DeformableParameterPointerType current, typename DeformableTransformType::DeformableParameterPointerType & next, typename DeformableTransformType::DeformableParameterPointerType currentFixed, typename DeformableTransformType::DeformableParameterPointerType & nextFixed)
{
  niftkitkDebugMacro(<< "CalculateNextStep():Started");
  typedef ImageRegionIterator<typename DeformableTransformType::DeformableParameterType> DeformableParametersIteratorType; 
  typedef IdentityTransform<double, Dimension> IdentityTransformType;
  typedef ImageRegionIterator<StepSizeImageType> StepSizeImageIteratorType; 
  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();
  double bestSimilarity = 0.; 
  
  if (m_ForceFilter.IsNull())
  {
      itkExceptionMacro(<< "Force filter is not set");
  } 
  if (m_FluidPDESolver.IsNull())
  {
      itkExceptionMacro(<< "PDE filter is not set");
  } 
  
  typename Superclass::ResampleFilterType::Pointer fixedImageRegriddingResampler = Superclass::ResampleFilterType::New(); 
  typename Superclass::ResampleFilterType::Pointer movingImageRegriddingResampler = Superclass::ResampleFilterType::New(); 
  
  fixedImageRegriddingResampler->SetInput(this->m_FixedImage);
  fixedImageRegriddingResampler->SetTransform(this->m_DeformableTransform);
  fixedImageRegriddingResampler->SetInterpolator(this->m_FixedImageInterpolator);      
  fixedImageRegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
  if (TFixedImage::ImageDimension == 3)
    fixedImageRegriddingResampler->SetDefaultPixelValue(this->m_RegriddedMovingImagePadValue);
  else
    fixedImageRegriddingResampler->SetDefaultPixelValue(254);
  
  movingImageRegriddingResampler->SetInput(this->m_MovingImage);
  movingImageRegriddingResampler->SetTransform(this->m_DeformableTransform);
  movingImageRegriddingResampler->SetInterpolator(this->m_MovingImageInterpolator);      
  movingImageRegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
  if (TFixedImage::ImageDimension == 3)
    movingImageRegriddingResampler->SetDefaultPixelValue(this->m_RegriddedMovingImagePadValue);
  else
    movingImageRegriddingResampler->SetDefaultPixelValue(254);
  
  // Debug images. 
  typedef Image<unsigned char, 2> OutputImageType2D; 
  typedef RescaleIntensityImageFilter<TFixedImage, OutputImageType2D> RescalerType;
  typename ImageFileWriter<OutputImageType2D>::Pointer writer2D = ImageFileWriter<OutputImageType2D>::New(); 
  typename RescalerType::Pointer intensityRescaler = RescalerType::New();
  typedef Image<float, TFixedImage::ImageDimension> OutputImageType; 
  typename ImageFileWriter<typename Superclass::ResampleFilterType::OutputImageType>::Pointer writer = ImageFileWriter<typename Superclass::ResampleFilterType::OutputImageType>::New(); 
  // char filename[1000]; 
  intensityRescaler->SetOutputMinimum(0);
  intensityRescaler->SetOutputMaximum(255);
  writer2D->SetInput(intensityRescaler->GetOutput()); 
  // Debug images. 
  
  VelocityFieldTransformType* transform = GetMovingImageVelocityFieldDeformableTransform(); 
  // typename std::vector<typename DeformableParameterType::Pointer> nextVelocityField; 
  // typename std::vector<typename DeformableParameterType::Pointer> currentVelocityFieldVector; 
  
  bool isSmallEnoughDeformation = true; 
  
  // transform->ResetDeformationFields(); 
  // transform->AccumulateDeformationFromVelocityField(transform->GetNumberOfVelocityField()); 
  
  // Go through and update all the velocity field using the current deformation at each time point. 
  for (int i = 0; i < transform->GetNumberOfVelocityField(); i++)
  {
    niftkitkDebugMacro(<< "CalculateNextStep():Update time point " << i);
    
    // Accumulate the deformation field from the velocity field. 
    // transform->AccumulateDeformationFromVelocityField(i); 
    //transform->AccumulateBackwardDeformationFromVelocityField(i); 
    //transform->AccumulateForwardDeformationFromVelocityField(i);   
    
    // Transform the fixed and moving images. 
    fixedImageRegriddingResampler->Modified(); 
    transform->LoadFixedImageDeformationField(i); 
    transform->ComputeMinJacobian(); 
    typename VelocityFieldTransformType::JacobianDeterminantFilterType::OutputImageType::Pointer fixedImageTransformJacobian = transform->GetJacobianImage(); 
    fixedImageTransformJacobian->DisconnectPipeline(); 
    fixedImageRegriddingResampler->UpdateLargestPossibleRegion(); 
    
#if 0    
    sprintf(filename, "fixed_%d_%d.png", this->m_CurrentIteration, i); 
    intensityRescaler->SetInput(fixedImageRegriddingResampler->GetOutput()); 
    writer->SetFileName(filename); 
    writer->Update(); 
    sprintf(filename, "jac_fixed_%d_%d.png", this->m_CurrentIteration, i); 
    writer->SetFileName(filename); 
    intensityRescaler->SetInput(fixedImageTransformJacobian); 
    writer->Update(); 
#endif                          
    
    movingImageRegriddingResampler->Modified(); 
    transform->LoadMovingImageDeformationField(i); 
    transform->ComputeMinJacobian(); 
    typename VelocityFieldTransformType::JacobianDeterminantFilterType::OutputImageType::Pointer movingImageTransformJacobian = transform->GetJacobianImage(); 
    movingImageTransformJacobian->DisconnectPipeline(); 
    movingImageRegriddingResampler->UpdateLargestPossibleRegion(); 
    
#if 0    
    sprintf(filename, "moving_%d_%d.png", this->m_CurrentIteration, i); 
    intensityRescaler->SetInput(movingImageRegriddingResampler->GetOutput()); 
    writer->SetFileName(filename); 
    writer->Update(); 
    sprintf(filename, "jac_moving_%d_%d.png", this->m_CurrentIteration, i); 
    writer->SetFileName(filename); 
    intensityRescaler->SetInput(movingImageTransformJacobian); 
    writer->Update(); 
#endif
    
    this->m_ForceFilter->SetIsSymmetric(true); 
    this->m_ForceFilter->SetFixedImage(fixedImageRegriddingResampler->GetOutput());
    this->m_ForceFilter->SetTransformedMovingImage(movingImageRegriddingResampler->GetOutput());
    this->m_ForceFilter->SetUnTransformedMovingImage(movingImageRegriddingResampler->GetOutput());
    this->m_ForceFilter->SetFixedImageMask(this->m_ImageToImageMetric->GetFixedImageMask()); 
    this->m_ForceFilter->SetFixedImageTransformJacobian(movingImageTransformJacobian); 
    this->m_ForceFilter->SetMovingImageTransformJacobian(fixedImageTransformJacobian); 
    this->m_ForceFilter->UpdateLargestPossibleRegion(); 
    //niftkitkDebugMacro(<< "CalculateNextStep():Updating force");
    this->m_ForceFilter->UpdateLargestPossibleRegion();
    this->m_FluidPDESolver->SetInput(this->m_ForceFilter->GetOutput());
    //niftkitkDebugMacro(<< "CalculateNextStep():Updating solver");
    this->m_FluidPDESolver->UpdateLargestPossibleRegion(); 
    
    //typename DeformableParameterType::Pointer tempVelocityField = this->m_FluidPDESolver->GetOutput(); 
    //tempVelocityField->DisconnectPipeline(); 
    //this->m_FluidPDESolver->SetInput(tempVelocityField);
    //niftkitkDebugMacro(<< "CalculateNextStep():Updating solver");
    //this->m_FluidPDESolver->UpdateLargestPossibleRegion(); 
    
    typename DeformableParameterType::Pointer velocityField = this->m_FluidPDESolver->GetOutput(); 
    // typename DeformableParameterType::Pointer currentVelocityField = GetMovingImageVelocityFieldDeformableTransform()->GetVelocityField(i);
    GetMovingImageVelocityFieldDeformableTransform()->LoadVelocityField(i);
    typename DeformableParameterType::Pointer currentVelocityField = GetMovingImageVelocityFieldDeformableTransform()->GetVelocityField(0); 
    
    // Calculate the gradient/change in velcity field. 
    DeformableParametersIteratorType iterator(velocityField, velocityField->GetLargestPossibleRegion()); 
    DeformableParametersIteratorType currentIterator(currentVelocityField, currentVelocityField->GetLargestPossibleRegion()); 
    double maxNorm = 0.; 
    double maxCurrentNorm = 0.; 
    double maxChange = 0.;
      
    for (iterator.GoToBegin(), currentIterator.GoToBegin(); 
         !iterator.IsAtEnd(); 
         ++iterator, ++currentIterator)
    {
      typename DeformableTransformType::DeformableParameterType::PixelType value = iterator.Get(); 
      typename DeformableTransformType::DeformableParameterType::PixelType currentValue = currentIterator.Get(); 
      double norm = value.GetNorm(); 
      double currentNorm = currentValue.GetNorm(); 
      typename DeformableTransformType::DeformableParameterType::PixelType diff = value-currentValue; 
      double change = diff.GetNorm(); ; 
      
      if (norm > maxNorm)
        maxNorm = norm;  
      if (currentNorm > maxCurrentNorm)
        maxCurrentNorm = currentNorm;
      if (change > maxChange)
        maxChange = change; 
    }
    if (maxCurrentNorm == 0)
      maxCurrentNorm = 1; 
    niftkitkDebugMacro(<< "CalculateNextStep():maxChange=" << maxChange << ",maxNorm=" << maxNorm << ",maxCurrentNorm=" << maxCurrentNorm );
    // niftkitkDebugMacro(<< "CalculateNextStep():stepSize=" << this->m_StepSize);
    
    typename DeformableParameterType::Pointer currentVelocityFieldGradient = DeformableTransformType::DuplicateDeformableParameters(GetMovingImageVelocityFieldDeformableTransform()->GetVelocityField(0)); 
    DeformableParametersIteratorType currentVelocityFieldGradientIterator(currentVelocityFieldGradient, currentVelocityFieldGradient->GetLargestPossibleRegion());  
    
    LoadStepSizeImage(this->m_StepSizeImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeImageFilename(i)); 
    StepSizeImageIteratorType stepSizeImageIterator(this->m_StepSizeImage[0], this->m_StepSizeImage[0]->GetLargestPossibleRegion()); 
    
    double maxDeformationChange = 0.; 
    // Update the velocity field with the change. 
    for (iterator.GoToBegin(), currentIterator.GoToBegin(), currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin();  
         !iterator.IsAtEnd(); 
         ++iterator, ++currentIterator, ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator)
    {
      typename DeformableTransformType::DeformableParameterType::PixelType value = iterator.Get(); 
      typename DeformableTransformType::DeformableParameterType::PixelType currentValue = currentIterator.Get(); 
      typename DeformableTransformType::DeformableParameterType::PixelType gradient = value - currentValue; 
      double stepSize = stepSizeImageIterator.Get();
      // double stepSize = this->m_StepSize; 
      
      double maxAllowedFactor = 1.; 
      if (maxChange > stepSize)
      {
        maxAllowedFactor = stepSize/maxChange; 
      }
      for (unsigned int j = 0; j < Dimension; j++)
      {
        double change = (value[j]-currentValue[j])*maxAllowedFactor; 
        currentValue[j] = currentValue[j] + change;
        maxDeformationChange = std::max<double>(maxDeformationChange, fabs(change)); 
      }
      iterator.Set(currentValue); 
      currentVelocityFieldGradientIterator.Set(gradient); 
    }
    niftkitkDebugMacro(<< "CalculateNextStep():maxDeformationChange=" << maxDeformationChange);
    
    if (maxDeformationChange > this->m_MinimumDeformationMagnitudeThreshold)
    {
      isSmallEnoughDeformation = false; 
    }
    
    // velocityField->DisconnectPipeline(); 
    transform->SaveVelocityField(velocityField, i); 
    // nextVelocityField.push_back(velocityField); 
    // currentVelocityFieldVector.push_back(transform->GetVelocityField(i)); 
    
    //(const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetFixedImage(fixedImageRegriddingResampler->GetOutput());
    //(const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetMovingImage(movingImageRegriddingResampler->GetOutput());
    //(const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetTransform(identityTransform); 
    //double similarity = this->m_ImageToImageMetric->GetValue(this->m_DeformableTransform->GetParameters()); 
    //bestSimilarity += similarity;
    //niftkitkDebugMacro(<< "CalculateNextStep():Current similarity=" << similarity);
    
    DeformableTransformType::LoadField(&(this->m_PreviousVelocityFieldGradient[0]), VelocityFieldDeformableTransformFilename::GetPreviousVelocityFieldGradientFilename(i)); 
    DeformableParametersIteratorType preVelocityFieldGradientIterator(this->m_PreviousVelocityFieldGradient[0], this->m_PreviousVelocityFieldGradient[0]->GetLargestPossibleRegion());  
    LoadStepSizeImage(this->m_StepSizeNormalisationFactorImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeNormalisationFactorFilename(i)); 
    StepSizeImageIteratorType stepSizeNormalisationIteration(this->m_StepSizeNormalisationFactorImage[0], this->m_StepSizeNormalisationFactorImage[0]->GetLargestPossibleRegion()); 
    
#if 1    
#if 1    
    for (currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin(), preVelocityFieldGradientIterator.GoToBegin(), stepSizeNormalisationIteration.GoToBegin();  
         !currentVelocityFieldGradientIterator.IsAtEnd(); 
         ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator, ++preVelocityFieldGradientIterator, ++stepSizeNormalisationIteration)
    {
      float stepSize = stepSizeImageIterator.Get();
      float normalisationFactor = stepSizeNormalisationIteration.Get(); 
      const double mu = 0.9; 
      const double gamma = 0.2; 
      
      normalisationFactor = mu*normalisationFactor + (currentVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get())*(1.-mu); 
      double stepSizeFactor = 1. + (preVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get())*gamma/normalisationFactor; 
      stepSize = stepSize*std::min(2., std::max(0.5, stepSizeFactor)); 
      
      if (stepSize > this->m_InitialStepSize)
        stepSize = this->m_InitialStepSize; 
      
      stepSizeNormalisationIteration.Set(normalisationFactor); 
      stepSizeImageIterator.Set(stepSize); 
    }
    SaveStepSizeImage(this->m_StepSizeImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeImageFilename(i)); 
    SaveStepSizeImage(this->m_StepSizeNormalisationFactorImage[0], VelocityFieldDeformableTransformFilename::GetStepSizeNormalisationFactorFilename(i)); 
    
    
#else
    double prevCurrentNorm = 0.; 
    double currentNorm = 0.; 
    for (currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin(), preVelocityFieldGradientIterator.GoToBegin(), stepSizeNormalisationIteration.GoToBegin();  
         !currentVelocityFieldGradientIterator.IsAtEnd(); 
         ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator, ++preVelocityFieldGradientIterator, ++stepSizeNormalisationIteration)
    {
      currentNorm += currentVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get(); 
      prevCurrentNorm += preVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get(); 
    }
    for (currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin(), preVelocityFieldGradientIterator.GoToBegin(), stepSizeNormalisationIteration.GoToBegin();  
         !currentVelocityFieldGradientIterator.IsAtEnd(); 
         ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator, ++preVelocityFieldGradientIterator, ++stepSizeNormalisationIteration)
    {
      float stepSize = stepSizeImageIterator.Get();
      float normalisationFactor = stepSizeNormalisationIteration.Get(); 
      const double mu = 0.9; 
      const double gamma = 0.01; 
      
      normalisationFactor = mu*normalisationFactor + currentNorm*(1.-mu); 
      stepSize = stepSize*std::max(0.5, (1. + prevCurrentNorm*gamma/normalisationFactor)); 
      
      if (stepSize > this->m_InitialStepSize)
        stepSize = this->m_InitialStepSize; 
      
      stepSizeNormalisationIteration.Set(normalisationFactor); 
      stepSizeImageIterator.Set(stepSize); 
    }
#endif    
#endif    
    // this->m_PreviousVelocityFieldGradient[i] = currentVelocityFieldGradient; 
    DeformableTransformType::SaveField(currentVelocityFieldGradient, VelocityFieldDeformableTransformFilename::GetPreviousVelocityFieldGradientFilename(i)); 
  }
  
  //for (int i = 0; i < transform->GetNumberOfVelocityField(); i++)
  //{
  //  GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(nextVelocityField[i], i); 
  //}
  // checking.
  double minJacobian = 0.;   
  
  //transform->AccumulateDeformationFromVelocityField(0); 
  //transform->UseFixedImageDeformationField(); 
  //minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  //transform->UseMovingImageDeformationField(); 
  //minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  
  transform->AccumulateDeformationFromVelocityField(1); 
  transform->AccumulateDeformationFromVelocityField(transform->GetNumberOfVelocityField()); 
  transform->LoadFixedImageDeformationField(0); 
  minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  fixedImageRegriddingResampler->Modified(); 
  fixedImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  movingImageRegriddingResampler->Modified(); 
  transform->LoadMovingImageDeformationField(transform->GetNumberOfVelocityField()-1); 
  next = transform->GetDeformableParameters(); 
  minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  movingImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetFixedImage(this->m_FixedImage);
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetMovingImage(movingImageRegriddingResampler->GetOutput());
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetTransform(identityTransform); 
  double similarity = this->m_ImageToImageMetric->GetValue(this->m_DeformableTransform->GetParameters()); 
  bestSimilarity += similarity;
  niftkitkDebugMacro(<< "CalculateNextStep():Current similarity=" << similarity);
  
#if 0    
  if (TFixedImage::ImageDimension == 2)
  {
    sprintf(filename, "moving_%d_20.png", this->m_CurrentIteration); 
    intensityRescaler->SetInput(movingImageRegriddingResampler->GetOutput()); 
    writer2D->SetFileName(filename); 
    writer2D->Update(); 
  }
  else
  {
    sprintf(filename, "moving_%d_20.hdr", this->m_CurrentIteration); 
    writer->SetInput(movingImageRegriddingResampler->GetOutput()); 
    writer->SetFileName(filename); 
    writer->Update(); 
    sprintf(filename, "jac_moving_%d_20.str", this->m_CurrentIteration); 
    int origin[3] = { 0., 0., 0. }; 
    transform->WriteMidasStrImage(filename, origin, transform->GetJacobianImage()->GetLargestPossibleRegion(), transform->GetJacobianImage()); 
  }
#endif                          
                          
#if 0    
  typename TMovingImage::Pointer checkerBoard = TMovingImage::New();
  checkerBoard->SetRegions(this->m_MovingImage->GetLargestPossibleRegion().GetSize());
  checkerBoard->Allocate();
  for (unsigned int x = 0; x < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[0]; x++)
  {
    for (unsigned int y = 0; y < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[1]; y++)
    {
      typename TMovingImage::IndexType index; 
      index[0] = x;
      index[1] = y;
      checkerBoard->SetPixel(index, 200);
      if (x % 5 == 0 || y % 5 == 0)
      {
        checkerBoard->SetPixel(index, 0);
      }
    }
  }
  movingImageRegriddingResampler->SetInput(checkerBoard); 
  movingImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  sprintf(filename, "checker_%d_20.png", this->m_CurrentIteration); 
  intensityRescaler->SetInput(movingImageRegriddingResampler->GetOutput()); 
  writer->SetFileName(filename); 
  writer->Update(); 
    
#endif  
  
  //bestSimilarity /= (transform->GetNumberOfVelocityField()+1.); 
  
  // seems that using the final images is the best for adjusting the step size. 
  bestSimilarity = similarity; 
  
  
#if 0  
#if 0  
  if (minJacobian < 0.)
  {
    for (int i = 0; i < transform->GetNumberOfVelocityField(); i++)
    {
      GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(currentVelocityFieldVector[i], i); 
    }
    this->m_StepSize *= this->m_IteratingStepSizeReductionFactor;
    this->m_Value = 0.;
  }
  else 
#endif  
  if ((this->m_Maximize && bestSimilarity > this->m_Value) ||  (!this->m_Maximize && bestSimilarity < this->m_Value))
  {
    for (int i = 0; i < transform->GetNumberOfVelocityField(); i++)
    {
      GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(nextVelocityField[i], i); 
    }
    
    this->m_StepSize *= 1.1; 
    if (this->m_StepSize > this->m_InitialStepSize)
      this->m_StepSize = this->m_InitialStepSize; 
    
    // not sure exactly how to do this. dilating the time to keep the velocity constant for now. 
    transform->ReparameteriseTime(); 
  }
#if 1
  else
  {
    //this->m_StepSize *= this->m_IteratingStepSizeReductionFactor;
    this->m_StepSize *= 0.9;
    
    for (int i = 0; i < transform->GetNumberOfVelocityField(); i++)
    {
      GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(currentVelocityFieldVector[i], i); 
    }
    transform->AccumulateDeformationFromVelocityField(0); 
    transform->AccumulateDeformationFromVelocityField(transform->GetNumberOfVelocityField()); 
    // transform->ReparameteriseTime(); 
  }
#endif  
  
  //if ((this->m_CurrentIteration % 10) == 0)
  //{
    //transform->ReparameteriseTime(); 
  //}
#endif  
                    
  if (isSmallEnoughDeformation)
  {
    niftkitkDebugMacro(<< "CalculateNextStep():small enough deformation");
    this->m_StepSize = 0.; 
  }
  
  niftkitkDebugMacro(<< "CalculateNextStep():Finished");
  return bestSimilarity; 
}
#endif


template < typename TFixedImage, typename TMovingImage, typename TScalarType, class TDeformationScalar>
void
VelocityFieldGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::ResumeOptimization(void)
{
  niftkitkInfoMacro(<< "ResumeOptimization():Started");
  ParametersType dummyParameters; 
  this->m_InitialStepSize = this->m_StepSize;
  double minJacobian = 0;
  
  this->m_CurrentIteration = 0;

  niftkitkInfoMacro(<< "ResumeOptimization():Starting with initial value of:" << niftk::ConvertToString(this->m_Value));
  
  // Stop can be set by another thread WHILE we are iterating (potentially).
  this->m_Stop = false;
  
  // Subclasses can initialize at this point
  this->Initialize();
  
  // Save the starting value as the best. 
#ifndef SHOOT  
  DeformableTransformType::SaveField(GetMovingImageVelocityFieldDeformableTransform()->GetDeformableParameters(), VelocityFieldDeformableTransformFilename::GetBestDeformationFilename()); 
  GetMovingImageVelocityFieldDeformableTransform()->SaveBestVelocityField(); 
#endif  
  
  double nextValue = std::numeric_limits<double>::max(); 
  
  while( !this->m_Stop ) 
  {
    this->m_CurrentIteration++;
    
    double maxDeformationChange = 0.0;
  
    if (this->m_CurrentIteration > this->m_MaximumNumberOfIterations)
    {
      niftkitkInfoMacro(<< "ResumeOptimization():Hit max iterations:" << this->m_MaximumNumberOfIterations);
      this->StopOptimization();
      break;
    }
    
    if (this->m_StepSize < this->m_MinimumStepSize)
    {
      niftkitkInfoMacro(<< "ResumeOptimization():Gone below min step size:" << this->m_MinimumStepSize);
      this->StopOptimization();
      break;
    }

    // Calculate the next set of parameters, which may or may not be 'accepted'.
    nextValue = CalculateNextStep(this->m_CurrentIteration, this->m_Value, this->m_CurrentDeformableParameters, this->m_NextDeformableParameters, this->m_CurrentFixedDeformableParameters, this->m_NextFixedDeformableParameters);
    
    // So far so good, so compute new similarity measure.                                                                               
    if (nextValue == std::numeric_limits<double>::max())
    {
      nextValue = this->GetCostFunction()->GetValue(dummyParameters);
    }

#if 0    
    // Calculate the max deformation change. 
    typedef ImageRegionConstIterator<typename DeformableTransformType::DeformableParameterType> DeformableParametersIteratorType; 
    DeformableParametersIteratorType currentDeformableParametersIterator(this->m_CurrentDeformableParameters, this->m_CurrentDeformableParameters->GetLargestPossibleRegion()); 
    DeformableParametersIteratorType nextDeformableParametersIterator(this->m_NextDeformableParameters, this->m_NextDeformableParameters->GetLargestPossibleRegion()); 
      
    for (currentDeformableParametersIterator.GoToBegin(), nextDeformableParametersIterator.GoToBegin(); 
         !currentDeformableParametersIterator.IsAtEnd(); 
         ++currentDeformableParametersIterator, ++nextDeformableParametersIterator)
    {
      for (unsigned int i = 0; i < Dimension; i++)
        {
          double change = fabs(currentDeformableParametersIterator.Get()[i]-nextDeformableParametersIterator.Get()[i]);
          if (change > maxDeformationChange)
            {
              maxDeformationChange = change;  
            }
        }
    }
      
    niftkitkDebugMacro(<< "ResumeOptimization(): maxDeformationChange=" << maxDeformationChange << " and m_MinimumDeformationMagnitudeThreshold=" << this->m_MinimumDeformationMagnitudeThreshold << ", and m_CheckMinDeformationMagnitudeThreshold=" << this->m_CheckMinDeformationMagnitudeThreshold);
#endif    
    
#if 0    
    if (this->m_CheckMinDeformationMagnitudeThreshold && maxDeformationChange < this->m_MinimumDeformationMagnitudeThreshold)
    {
      niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << this->m_CurrentIteration \
        << ", maxDeformation was:" << maxDeformationChange \
        << ", threshold was:" << this->m_MinimumDeformationMagnitudeThreshold \
        << ", so I'm rejecting this change, reducing step size and continuing");

      this->m_StepSize *= this->m_IteratingStepSizeReductionFactor;
    }
    else
#endif                                   
    {
      niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << this->m_CurrentIteration \
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
        
#ifndef SHOOT        
        DeformableTransformType::SaveField(this->m_NextDeformableParameters, VelocityFieldDeformableTransformFilename::GetBestDeformationFilename()); 
        GetMovingImageVelocityFieldDeformableTransform()->SaveBestVelocityField(); 
#else        
        GetMovingImageVelocityFieldDeformableTransform()->UseMovingImageDeformationField(); 
        this->m_CurrentDeformableParameters = DeformableTransformType::DuplicateDeformableParameters(GetMovingImageVelocityFieldDeformableTransform()->GetDeformableParameters()); 
        //GetMovingImageVelocityFieldDeformableTransform()->UseFixedImageDeformationField(); 
        //this->m_CurrentFixedDeformableParameters = DeformableTransformType::DuplicateDeformableParameters(GetMovingImageVelocityFieldDeformableTransform()->GetDeformableParameters()); 
#endif        
        this->m_BestIteration = this->m_CurrentIteration; 
      
        if (fabs(nextValue - this->m_Value) <  this->m_MinimumSimilarityChangeThreshold
            && this->m_CheckSimilarityMeasure)
        {
          niftkitkInfoMacro(<< "ResumeOptimization():Iteration=" << this->m_CurrentIteration \
            << ", similarity change was:" << fabs(nextValue - this->m_Value) \
            << ", threshold was:" << this->m_MinimumSimilarityChangeThreshold \
            << ", so I'm accepting this change, but then stopping.");
          // this->m_CurrentDeformableParameters = DeformableTransformType::DuplicateDeformableParameters(this->m_NextDeformableParameters); 
          this->m_Value = nextValue;
          this->StopOptimization();
          break;
        }
        this->m_Value = nextValue;
      }
      else
      {
        //this->m_StepSize *= this->m_IteratingStepSizeReductionFactor;

        niftkitkInfoMacro(<< "ResumeOptimization():Maximize:" << this->m_Maximize \
          << ", currentValue:" << niftk::ConvertToString(this->m_Value) \
          << ", nextValue:" << niftk::ConvertToString(nextValue) \
          << ", so its no better, so rejecting this change, reducing step size by:" \
          << this->m_IteratingStepSizeReductionFactor \
          << ", to:" << this->m_StepSize \
          << " and continuing");
        
        if (this->m_CurrentIteration - this->m_BestIteration > this->m_WorseNumberOfIterationsAllowed)
        {
          niftkitkInfoMacro(<< "ResumeOptimization():Maximize: cannot find better similarity in " << this->m_WorseNumberOfIterationsAllowed << " iterations");
          break; 
        }
      }
    } 

  } // End main while loop.

  // Subclasses can cleanup at this point
  this->CleanUp();
  
  // Im resetting the m_Step size, so that if you repeatedly call 
  // the optimizer, it does actually do some optimizing.
  this->m_StepSize = this->m_InitialStepSize;
  
#ifndef SHOOT        
  DeformableTransformType::LoadField(&(this->m_CurrentDeformableParameters), VelocityFieldDeformableTransformFilename::GetBestDeformationFilename()); 
#endif  
  GetMovingImageVelocityFieldDeformableTransform()->SetDeformableParameters(this->m_CurrentDeformableParameters); 
  
  niftkitkInfoMacro(<< "ResumeOptimization():Finished");
}

#if 1
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
double
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::CalculateNextStep(int iterationNumber, double currentSimilarity, typename DeformableTransformType::DeformableParameterPointerType current, typename DeformableTransformType::DeformableParameterPointerType & next, typename DeformableTransformType::DeformableParameterPointerType currentFixed, typename DeformableTransformType::DeformableParameterPointerType & nextFixed)
{
  niftkitkDebugMacro(<< "CalculateNextStep():Started");
  typedef ImageRegionIterator<typename DeformableTransformType::DeformableParameterType> DeformableParametersIteratorType; 
  typedef ImageRegionIterator<TFixedImage> StepSizeIteratorType; 
  typedef IdentityTransform<double, Dimension> IdentityTransformType;
  typedef ImageRegionIterator<StepSizeImageType> StepSizeImageIteratorType; 
  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();
  double bestSimilarity = 0.; 
  
  if (m_ForceFilter.IsNull())
  {
      itkExceptionMacro(<< "Force filter is not set");
  } 
  if (m_FluidPDESolver.IsNull())
  {
      itkExceptionMacro(<< "PDE filter is not set");
  } 
  
  typename Superclass::ResampleFilterType::Pointer fixedImageRegriddingResampler = Superclass::ResampleFilterType::New(); 
  typename Superclass::ResampleFilterType::Pointer movingImageRegriddingResampler = Superclass::ResampleFilterType::New(); 
  
  fixedImageRegriddingResampler->SetInput(this->m_FixedImage);
  fixedImageRegriddingResampler->SetTransform(this->m_DeformableTransform);
  fixedImageRegriddingResampler->SetInterpolator(this->m_FixedImageInterpolator);      
  fixedImageRegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
  // fixedImageRegriddingResampler->SetDefaultPixelValue(this->m_RegriddedMovingImagePadValue);
  fixedImageRegriddingResampler->SetDefaultPixelValue(254);
  
  movingImageRegriddingResampler->SetInput(this->m_MovingImage);
  movingImageRegriddingResampler->SetTransform(this->m_DeformableTransform);
  movingImageRegriddingResampler->SetInterpolator(this->m_MovingImageInterpolator);      
  movingImageRegriddingResampler->SetOutputParametersFromImage(this->m_FixedImage);
  // movingImageRegriddingResampler->SetDefaultPixelValue(this->m_RegriddedMovingImagePadValue);
  movingImageRegriddingResampler->SetDefaultPixelValue(254);
  
  // Debug images. 
  typedef Image<unsigned char, TFixedImage::ImageDimension> OutputImageType; 
  typename ImageFileWriter<OutputImageType>::Pointer writer = ImageFileWriter<OutputImageType>::New(); 
  typedef RescaleIntensityImageFilter<TMovingImage, OutputImageType>   RescalerType;
  typename RescalerType::Pointer intensityRescaler = RescalerType::New();
  char filename[1000]; 
  intensityRescaler->SetOutputMinimum(0);
  intensityRescaler->SetOutputMaximum(255);
  writer->SetInput(intensityRescaler->GetOutput()); 
  // Debug images. 
  
  VelocityFieldTransformType* transform = GetMovingImageVelocityFieldDeformableTransform(); 
  typename std::vector<typename DeformableParameterType::Pointer> nextVelocityField; 
  typename std::vector<typename DeformableParameterType::Pointer> currentVelocityFieldVector; 
  
  // Transform the fixed images. 
  fixedImageRegriddingResampler->Modified(); 
  transform->UseFixedImageDeformationField(); 
  //transform->ComputeMinJacobian(); 
  //typename VelocityFieldTransformType::JacobianDeterminantFilterType::OutputImageType::Pointer fixedImageTransformJacobian = transform->GetJacobianImage(); 
  //fixedImageTransformJacobian->DisconnectPipeline(); niftkitkDebugMacro
  fixedImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  
#if 0
  sprintf(filename, "fixed_%d_%d.png", this->m_CurrentIteration, 0); 
  intensityRescaler->SetInput(fixedImageRegriddingResampler->GetOutput()); 
  writer->SetFileName(filename); 
  writer->Update(); 
  sprintf(filename, "jac_fixed_%d_%d.png", this->m_CurrentIteration, 0); 
  if (transform->GetForwardJacobianImage().IsNotNull())
  {
    writer->SetFileName(filename); 
    intensityRescaler->SetInput(transform->GetForwardJacobianImage()); 
    writer->Update(); 
  }
#endif                        
  
  // Momemtum of the backward transform. 
  this->m_ForceFilter->SetIsSymmetric(false); 
  this->m_ForceFilter->SetFixedImage(fixedImageRegriddingResampler->GetOutput());
  this->m_ForceFilter->SetTransformedMovingImage(this->m_MovingImage);
  this->m_ForceFilter->SetUnTransformedMovingImage(this->m_MovingImage);
  this->m_ForceFilter->SetFixedImageMask(this->m_ImageToImageMetric->GetFixedImageMask()); 
  // this->m_ForceFilter->SetFixedImageTransformJacobian(movingImageTransformJacobian); 
  // this->m_ForceFilter->SetFixedImageTransformJacobian(transform->GetBackwardJacobianImage()); 
  // this->m_ForceFilter->SetMovingImageTransformJacobian(transform->GetForwardJacobianImage()); 
  this->m_ForceFilter->UpdateLargestPossibleRegion(); 
  this->m_ForceFilter->UpdateLargestPossibleRegion();
  this->m_FluidPDESolver->SetInput(this->m_ForceFilter->GetOutput());
  this->m_FluidPDESolver->UpdateLargestPossibleRegion(); 
  typename DeformableParameterType::Pointer currentVelocityField = this->m_FluidPDESolver->GetOutput(); 
  currentVelocityField->DisconnectPipeline(); 
  
  typename DeformableParameterType::Pointer prevVelocityField = DeformableTransformType::DuplicateDeformableParameters(GetMovingImageVelocityFieldDeformableTransform()->GetVelocityField(0)); 
  typename DeformableParameterType::Pointer currentVelocityFieldGradient = DeformableTransformType::DuplicateDeformableParameters(GetMovingImageVelocityFieldDeformableTransform()->GetVelocityField(0)); 
  
  // Calculate the gradient/change in velcity field. 
  DeformableParametersIteratorType iterator(prevVelocityField, prevVelocityField->GetLargestPossibleRegion()); 
  DeformableParametersIteratorType currentIterator(currentVelocityField, currentVelocityField->GetLargestPossibleRegion()); 
  DeformableParametersIteratorType currentVelocityFieldGradientIterator(currentVelocityFieldGradient, currentVelocityFieldGradient->GetLargestPossibleRegion());  
  double maxNorm = 0.; 
  double maxCurrentNorm = 0.; 
  double maxChange = 0.;
    
  for (iterator.GoToBegin(), currentIterator.GoToBegin(); 
        !iterator.IsAtEnd(); 
        ++iterator, ++currentIterator)
  {
    typename DeformableTransformType::DeformableParameterType::PixelType value = iterator.Get(); 
    typename DeformableTransformType::DeformableParameterType::PixelType currentValue = currentIterator.Get(); 
    double norm = value.GetNorm(); 
    double currentNorm = currentValue.GetNorm(); 
    typename DeformableTransformType::DeformableParameterType::PixelType diff = value-currentValue; 
    double change = diff.GetNorm(); ; 
    
    if (norm > maxNorm)
      maxNorm = norm;  
    if (currentNorm > maxCurrentNorm)
      maxCurrentNorm = currentNorm;
    if (change > maxChange)
      maxChange = change; 
  }
  if (maxCurrentNorm == 0)
    maxCurrentNorm = 1; 
  niftkitkInfoMacro(<< "CalculateNextStep():maxChange=" << maxChange << ",maxNorm=" << maxNorm << ",maxCurrentNorm=" << maxCurrentNorm );
  // niftkitkDebugMacro(<< "CalculateNextStep():stepSize=" << this->m_StepSize);
  
  double maxVeclocityChange = 0.; 
  
#ifndef SHOOT  
  StepSizeImageIteratorType stepSizeImageIterator(this->m_StepSizeImage[0], this->m_StepSizeImage[0]->GetLargestPossibleRegion()); 
      
  // Update the velocity field with the change. 
  for (iterator.GoToBegin(), currentIterator.GoToBegin(), currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin();  
         !iterator.IsAtEnd(); 
         ++iterator, ++currentIterator, ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator)
#else      
  for (iterator.GoToBegin(), currentIterator.GoToBegin(), currentVelocityFieldGradientIterator.GoToBegin();  
       !iterator.IsAtEnd(); 
       ++iterator, ++currentIterator, ++currentVelocityFieldGradientIterator)
#endif 
  {
    typename DeformableTransformType::DeformableParameterType::PixelType value = iterator.Get(); 
    typename DeformableTransformType::DeformableParameterType::PixelType currentValue = currentIterator.Get(); 
    typename DeformableTransformType::DeformableParameterType::PixelType gradient = value - currentValue; 
    //double stepSize = stepSizeImageIterator.Get();
    double stepSize = this->m_StepSize; 
    
    double maxAllowedFactor = 1.; 
    if (maxChange > stepSize)
    {
      maxAllowedFactor = stepSize/maxChange; 
    }
    typename DeformableTransformType::DeformableParameterType::PixelType change = (currentValue-value)*maxAllowedFactor; 
    maxVeclocityChange = std::max<double>(change.GetNorm(), maxVeclocityChange); 
    
    for (unsigned int j = 0; j < Dimension; j++)
    {
      // value[j] = value[j] + (currentValue[j]-value[j])*maxAllowedFactor;
      value[j] = value[j] + currentValue[j]*maxAllowedFactor;
    }
    
    currentIterator.Set(value); 
    currentVelocityFieldGradientIterator.Set(gradient); 
  }
  
  niftkitkInfoMacro(<< "CalculateNextStep():maxVeclocityChange=" << maxVeclocityChange);
  GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(currentVelocityField, 0); 
  GetMovingImageVelocityFieldDeformableTransform()->SetFluidPDESolver(this->m_FluidPDESolver); 
  GetMovingImageVelocityFieldDeformableTransform()->Shoot(); 
  
  // checking.
  double minJacobian = 0.;   
  
  //transform->AccumulateDeformationFromVelocityField(0); 
  //transform->UseFixedImageDeformationField(); 
  //minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  //transform->UseMovingImageDeformationField(); 
  //minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  
  transform->UseFixedImageDeformationField(); 
  minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  fixedImageRegriddingResampler->Modified(); 
  fixedImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  movingImageRegriddingResampler->Modified(); 
  transform->UseMovingImageDeformationField(); 
  minJacobian = std::min<double>(minJacobian, transform->ComputeMinJacobian()); 
  movingImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetFixedImage(this->m_FixedImage);
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetMovingImage(movingImageRegriddingResampler->GetOutput());
  //(const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetFixedImage(fixedImageRegriddingResampler->GetOutput());
  //(const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetMovingImage(this->m_MovingImage);
  (const_cast<ImageToImageMetricType*>(this->m_ImageToImageMetric))->SetTransform(identityTransform); 
  double similarity = this->m_ImageToImageMetric->GetValue(this->m_DeformableTransform->GetParameters()); 
  niftkitkInfoMacro(<< "CalculateNextStep():Current similarity=" << similarity);
  
  if (TFixedImage::ImageDimension == 2)
    sprintf(filename, "moving_%d_20.png", this->m_CurrentIteration); 
  else
    sprintf(filename, "moving_%d_20.hdr", this->m_CurrentIteration); 
  intensityRescaler->SetInput(movingImageRegriddingResampler->GetOutput()); 
  writer->SetFileName(filename); 
  writer->Update(); 
  
#if 0
  typename TMovingImage::Pointer checkerBoard = TMovingImage::New();
  checkerBoard->SetRegions(this->m_MovingImage->GetLargestPossibleRegion().GetSize());
  checkerBoard->Allocate();
  for (unsigned int x = 0; x < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[0]; x++)
  {
    for (unsigned int y = 0; y < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[1]; y++)
    {
      typename TMovingImage::IndexType index; 
      index[0] = x;
      index[1] = y;
      if (TFixedImage::ImageDimension == 2) 
      {
        checkerBoard->SetPixel(index, 200);
        if (x % 5 == 0 || y % 5 == 0)
        {
          checkerBoard->SetPixel(index, 0);
        }
      }
      if (TFixedImage::ImageDimension == 3) 
      {
        for (unsigned int z = 0; z < this->m_MovingImage->GetLargestPossibleRegion().GetSize()[2]; z++)
        {
          index[2] = z; 
          checkerBoard->SetPixel(index, 200);
          if (x % 5 == 0 || y % 5 == 0 || z % 5 == 0)
          {
            checkerBoard->SetPixel(index, 0);
          }
        }
      }
    }
  }
  transform->UseMovingImageDeformationField(); 
  movingImageRegriddingResampler->SetInput(checkerBoard); 
  movingImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  if (TFixedImage::ImageDimension == 2)
    sprintf(filename, "checker_%d_20.png", this->m_CurrentIteration); 
  else
    sprintf(filename, "checker_%d_20.hdr", this->m_CurrentIteration); 
  intensityRescaler->SetInput(movingImageRegriddingResampler->GetOutput()); 
  writer->SetFileName(filename); 
  writer->Update(); 
  transform->UseFixedImageDeformationField(); 
  fixedImageRegriddingResampler->SetInput(checkerBoard); 
  fixedImageRegriddingResampler->UpdateLargestPossibleRegion(); 
  if (TFixedImage::ImageDimension == 2)
    sprintf(filename, "checker_%d_20_fixed.png", this->m_CurrentIteration); 
  else
    sprintf(filename, "checker_%d_20_fixed.hdr", this->m_CurrentIteration); 
  intensityRescaler->SetInput(fixedImageRegriddingResampler->GetOutput()); 
  writer->SetFileName(filename); 
  writer->Update(); 
#endif  
  
#if 0
  //transform->ComputeMinJacobian(); 
  sprintf(filename, "jac_moving_%d_20.png", this->m_CurrentIteration); 
  writer->SetFileName(filename); 
  //intensityRescaler->SetInput(transform->GetJacobianImage()); 
  writer->Update(); 
#endif                        
  
  
  // seems that using the final images is the best for adjusting the step size. 
  bestSimilarity = similarity; 
  
  //if ((this->m_Maximize && bestSimilarity > this->m_Value) ||  (!this->m_Maximize && bestSimilarity < this->m_Value))
  //{
    // this->m_StepSize *= 1.25; 
  //  if (this->m_StepSize > this->m_InitialStepSize)
   //   this->m_StepSize = this->m_InitialStepSize; 
  //}
  //else
  //{
  //  this->m_StepSize *= this->m_IteratingStepSizeReductionFactor;
  //  GetMovingImageVelocityFieldDeformableTransform()->SetVelocityField(prevVelocityField, 0); 
  //}
  
#if 1
  DeformableParametersIteratorType preVelocityFieldGradientIterator(this->m_PreviousVelocityFieldGradient[0], this->m_PreviousVelocityFieldGradient[0]->GetLargestPossibleRegion());  
  double maxDotProduct = 0.; 
  double minDotProduct = 0.; 
  int numberOfVoxels = 0; 
  double meanDotProduct = 0.; 
  double meanCurrentVelcoityDotProduct = 0.; 
  if (this->m_CurrentIteration != 1)
  {
    for (currentVelocityFieldGradientIterator.GoToBegin(), preVelocityFieldGradientIterator.GoToBegin();  
         !currentVelocityFieldGradientIterator.IsAtEnd(); 
         ++currentVelocityFieldGradientIterator, ++preVelocityFieldGradientIterator)
    {
      double currentVelcoityDotProduct = currentVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get(); 
      double dotProduct = preVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get(); 
      
      meanCurrentVelcoityDotProduct = meanCurrentVelcoityDotProduct*numberOfVoxels/(numberOfVoxels+1.) + currentVelcoityDotProduct/(numberOfVoxels+1.); 
      meanDotProduct = meanDotProduct*numberOfVoxels/(numberOfVoxels+1.) + dotProduct/(numberOfVoxels+1.); 
      numberOfVoxels++; 
      maxDotProduct = std::max<double>(dotProduct, maxDotProduct); 
      minDotProduct = std::min<double>(dotProduct, minDotProduct); 
    }
  }
  const double mu = 0.9; 
  const double gamma = 0.05; 
  
  if (this->m_CurrentIteration == 1)
    this->m_NormalisationFactor = meanCurrentVelcoityDotProduct; 
  else
    this->m_NormalisationFactor = mu*this->m_NormalisationFactor + meanCurrentVelcoityDotProduct*(1.-mu); 
  
  double stepSizeFactor = 1. + meanDotProduct*gamma/this->m_NormalisationFactor; 
  if (this->m_CurrentIteration == 1)
    stepSizeFactor = 1.; 
  
  this->m_StepSize = this->m_StepSize*std::min(2., std::max(0.5, stepSizeFactor)); 
  niftkitkInfoMacro(<< "CalculateNextStep():this->m_StepSize=" << this->m_StepSize);
  if (this->m_StepSize > this->m_InitialStepSize)
    this->m_StepSize = this->m_InitialStepSize; 
  niftkitkInfoMacro(<< "CalculateNextStep():maxDotProduct=" << maxDotProduct << ",minDotProduct=" << minDotProduct << ",meanDotProduct=" << meanDotProduct);
  niftkitkInfoMacro(<< "CalculateNextStep():this->m_StepSize=" << this->m_StepSize);
  this->m_PreviousVelocityFieldGradient[0] = currentVelocityFieldGradient; 
  
#else
  DeformableParametersIteratorType preVelocityFieldGradientIterator(this->m_PreviousVelocityFieldGradient[0], this->m_PreviousVelocityFieldGradient[0]->GetLargestPossibleRegion());  
  StepSizeImageIteratorType stepSizeNormalisationIteration(this->m_StepSizeNormalisationFactorImage[0], this->m_StepSizeNormalisationFactorImage[0]->GetLargestPossibleRegion()); 
  for (currentVelocityFieldGradientIterator.GoToBegin(), stepSizeImageIterator.GoToBegin(), preVelocityFieldGradientIterator.GoToBegin(), stepSizeNormalisationIteration.GoToBegin();  
         !currentVelocityFieldGradientIterator.IsAtEnd(); 
         ++currentVelocityFieldGradientIterator, ++stepSizeImageIterator, ++preVelocityFieldGradientIterator, ++stepSizeNormalisationIteration)
    {
      float stepSize = stepSizeImageIterator.Get();
      float normalisationFactor = stepSizeNormalisationIteration.Get(); 
      const double mu = 0.9; 
      const double gamma = 0.01; 
      
      normalisationFactor = mu*normalisationFactor + (currentVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get())*(1.-mu); 
      double stepSizeFactor = 1. + (preVelocityFieldGradientIterator.Get()*currentVelocityFieldGradientIterator.Get())*gamma/normalisationFactor; 
      stepSize = stepSize*std::min(2., std::max(0.5, stepSizeFactor)); 
      
      if (stepSize > this->m_InitialStepSize)
        stepSize = this->m_InitialStepSize; 
      
      stepSizeNormalisationIteration.Set(normalisationFactor); 
      stepSizeImageIterator.Set(stepSize); 
    }
  this->m_PreviousVelocityFieldGradient[0] = currentVelocityFieldGradient; 
#endif                                                                
                                
  
  niftkitkDebugMacro(<< "CalculateNextStep():Finished");
  return bestSimilarity; 
}
#endif
                                
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
void
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::SaveStepSizeImage(typename StepSizeImageType::Pointer field, std::string filename)
{
  typedef ImageFileWriter<StepSizeImageType> ImageFileWriterType; 
  typename ImageFileWriterType::Pointer writer = ImageFileWriterType::New(); 
  
  writer->SetInput(field); 
  writer->SetFileName(filename); 
  writer->Update(); 
}

template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
void 
VelocityFieldGradientDescentOptimizer< TFixedImage, TMovingImage, TScalar, TDeformationScalar>
::LoadStepSizeImage(typename StepSizeImageType::Pointer& field, std::string filename)
{
  typedef ImageFileReader<StepSizeImageType> ImageFileReaderType; 
  typename ImageFileReaderType::Pointer reader = ImageFileReaderType::New(); 
  
  try 
  {
    reader->SetFileName(filename); 
    reader->Update(); 
    field = reader->GetOutput(); 
    field->DisconnectPipeline(); 
  }
  catch (itk::ExceptionObject &e)
  {
  }
}
                                


} // namespace itk.

#endif
