/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-26 10:49:56 +0100 (Thu, 26 May 2011) $
 Revision          : $Revision: 6271 $
 Last modified by  : $Author: kkl $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKVelocityFieldDeformableTransform_TXX_
#define ITKVelocityFieldDeformableTransform_TXX_

#include "itkVelocityFieldDeformableTransform.h" 
#include "itkVectorResampleImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkScaleTransform.h"
#include "itkIdentityTransform.h"
#include "itkImageDuplicator.h"
#include "itkVectorImageToImageAdaptor.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include <limits>
#include <sstream>
#include "itkDisplacementFieldJacobianFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkLogHelper.h"

namespace itk
{
// Constructor with default arguments
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::VelocityFieldDeformableTransform()
{
  niftkitkDebugMacro(<< "VelocityFieldDeformableTransform():Constructed");
  this->m_NumberOfVelocityField = 20; 
  return;
}

// Destructor
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::~VelocityFieldDeformableTransform()
{
  niftkitkDebugMacro(<< "VelocityFieldDeformableTransform():Destroyed");
  return;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  // Superclass one will do.
  Superclass::PrintSelf(os,indent);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{
  // Setup deformation field.
  Superclass::Initialize(image);
  
  this->m_VelocityField.clear(); 
  this->m_TimeStep.clear(); 
  
  // Copy image dimensions.
  typename DeformationFieldType::SpacingType spacing = image->GetSpacing();
  typename DeformationFieldType::DirectionType direction = image->GetDirection();
  typename DeformationFieldType::PointType origin = image->GetOrigin();
  typename DeformationFieldType::SizeType size = image->GetLargestPossibleRegion().GetSize();
  typename DeformationFieldType::IndexType index = image->GetLargestPossibleRegion().GetIndex();
  typename DeformationFieldType::RegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    typename DeformableParameterType::Pointer velocityField = DeformableParameterType::New(); 

    if (i == 0)
    {
      // And set them on our deformation field.
      velocityField->SetRegions(region);
      velocityField->SetOrigin(origin);
      velocityField->SetDirection(direction);
      velocityField->SetSpacing(spacing);
      velocityField->Allocate();
      velocityField->Update();
    
      // Set deformation field to zero.
      DeformationFieldPixelType fieldValue;
      fieldValue.Fill(0);
      velocityField->FillBuffer(fieldValue);
      m_VelocityField.push_back(velocityField); 
      // niftkitkDebugMacro(<< "region=" << region);
    }
    
    //SaveVelocityField(this->m_VelocityField[0], i); 
    
    //SaveField(this->m_DeformationField, FIXED_IMAGE_DEFROMATION_FILENAME, i); 
    //SaveField(this->m_DeformationField, MOVING_IMAGE_DEFROMATION_FILENAME, i); 
    
    m_TimeStep.push_back(1.); 
  }
  m_TimeStep.push_back(1.); 
  //SaveField(this->m_DeformationField, FIXED_IMAGE_DEFROMATION_FILENAME, this->m_NumberOfVelocityField); 
  //SaveField(this->m_DeformationField, MOVING_IMAGE_DEFROMATION_FILENAME, this->m_NumberOfVelocityField); 
  
  this->SetIdentity();   
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InitializeIdentityVelocityFields()
{
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    // niftkitkDebugMacro(<< "velocity field size=" << this->m_VelocityField[0]->GetLargestPossibleRegion());
    SaveVelocityField(this->m_VelocityField[0], i); 
    
    SaveField(this->m_DeformationField, VelocityFieldDeformableTransformFilename::GetFixedImageDeformationFilename(i)); 
    SaveField(this->m_DeformationField, VelocityFieldDeformableTransformFilename::GetMovingImageDeformationFieldFilename(i)); 
  }
  SaveField(this->m_DeformationField, VelocityFieldDeformableTransformFilename::GetFixedImageDeformationFilename(this->m_NumberOfVelocityField)); 
  SaveField(this->m_DeformationField, VelocityFieldDeformableTransformFilename::GetMovingImageDeformationFieldFilename(this->m_NumberOfVelocityField)); 
}



template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetIdentity( void )
{
  // This resets parameters and deformation field. Thats all we need.
  Superclass::SetIdentity();  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InterpolateNextGrid(FixedImagePointer image)
{
  niftkitkDebugMacro(<< "InterpolateNextGrid():Starting");
  
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    LoadVelocityField(i); 
    this->m_DeformationField = this->m_VelocityField[0]; 
    Superclass::InterpolateNextGrid(image); 
    this->m_VelocityField[0] = this->m_DeformationField; 
    SaveVelocityField(this->m_VelocityField[0], i); 
  }
  AccumulateDeformationFromVelocityField(1); 
  AccumulateDeformationFromVelocityField(this->m_NumberOfVelocityField); 
    
  niftkitkDebugMacro(<< "InterpolateNextGrid():Finished");
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::AccumulateDeformationFromVelocityField(int timePoint)
{
  niftkitkDebugMacro(<< "AccumulateDeformationFromVelocityField():timePoint=" << timePoint);
  typedef ImageRegionIterator<DeformableParameterType> DeformableParametersIteratorType; 
  typedef MultiplyImageFilter<typename Superclass::JacobianDeterminantFilterType::OutputImageType, typename Superclass::JacobianDeterminantFilterType::OutputImageType> MultiplyJacobianImageFilterType; 
  typename MultiplyJacobianImageFilterType::Pointer multiplyFilter = MultiplyJacobianImageFilterType::New(); 
  
  if (timePoint < -1 || timePoint > this->m_NumberOfVelocityField)
  {
    itkExceptionMacro(<< "AccumulateDeformationFromVelocityField: timePoint=" << timePoint << " can't be greater than this->m_NumberOfVelocityField=" << this->m_NumberOfVelocityField);
  }
  this->m_ForwardJacobianImage = NULL; 
  this->m_BackwardJacobianImage = NULL; 
  
  typename DeformationFieldType::PixelType fieldValue;
  fieldValue.Fill(0);
  this->m_DeformationField->FillBuffer(fieldValue);
  this->m_FixedImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
  this->m_MovingImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
  
#if 1
  
#if 0  
  for (int i = m_NumberOfVelocityField-1; i >= timePoint; i--)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
    this->m_DeformationField = temp; 
    DeformableParametersIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion()); 
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      iterator.Set(iterator.Get()*-1.); 
    }
    UpdateRegriddedDeformationParameters(this->m_FixedImageDeformationField, temp, this->m_TimeStep[i]); 
    SaveField(this->m_FixedImageDeformationField, FIXED_IMAGE_DEFROMATION_FILENAME, i); 
  }
  SaveField(this->m_MovingImageDeformationField, MOVING_IMAGE_DEFROMATION_FILENAME, 0); 
  for (int i = 0; i < timePoint; i++)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[i]); 
    
    SaveField(this->m_MovingImageDeformationField, MOVING_IMAGE_DEFROMATION_FILENAME, i+1); 
  }
#else  
  SaveField(this->m_FixedImageDeformationField, VelocityFieldDeformableTransformFilename::GetFixedImageDeformationFilename(m_NumberOfVelocityField-1)); 
  for (int i = m_NumberOfVelocityField-1; i >= timePoint; i--)
  {
    LoadVelocityField(i); 
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[0]); 
    this->m_DeformationField = temp; 
    DeformableParametersIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion()); 
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      iterator.Set(iterator.Get()*-1.); 
    }
    UpdateRegriddedDeformationParameters(this->m_FixedImageDeformationField, temp, this->m_TimeStep[i]); 
    SaveField(this->m_FixedImageDeformationField, VelocityFieldDeformableTransformFilename::GetFixedImageDeformationFilename(i-1)); 
  }
  for (int i = 0; i < timePoint; i++)
  {
    LoadVelocityField(i); 
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[0]); 
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[i]); 
    
    SaveField(this->m_MovingImageDeformationField, VelocityFieldDeformableTransformFilename::GetMovingImageDeformationFieldFilename(i)); 
  }
#endif                                
  
#else
                                                                
  // Playing around with a symmetric way to calculate the deformation. 
  niftkitkDebugMacro(<< "AccumulateDeformationFromVelocityField():accumulating fixed deformation field");
  typename DeformationFieldType::Pointer prevVelocityField = Superclass::DuplicateDeformableParameters(this->m_VelocityField[m_NumberOfVelocityField-1]); 
  
  for (int i = m_NumberOfVelocityField-1; i >= timePoint; i--)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
    DeformableParametersIteratorType prevIt(prevVelocityField, prevVelocityField->GetLargestPossibleRegion()); 
    DeformableParametersIteratorType currentIt(temp, temp->GetLargestPossibleRegion()); 
    for (prevIt.GoToBegin(), currentIt.GoToBegin(); 
         !prevIt.IsAtEnd(); 
         ++prevIt, ++currentIt)
    {
      currentIt.Set((prevIt.Get()+currentIt.Get())/(-2.)); 
    }
    
    UpdateRegriddedDeformationParameters(this->m_FixedImageDeformationField, temp, this->m_TimeStep[i+1]); 
    
    prevVelocityField = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
  }
  
  niftkitkDebugMacro(<< "AccumulateDeformationFromVelocityField():accumulating moving deformation field");
  prevVelocityField = Superclass::DuplicateDeformableParameters(this->m_VelocityField[0]); 
  for (int i = 0; i <= timePoint && i < m_NumberOfVelocityField; i++)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
    DeformableParametersIteratorType prevIt(prevVelocityField, prevVelocityField->GetLargestPossibleRegion()); 
    DeformableParametersIteratorType currentIt(temp, temp->GetLargestPossibleRegion()); 
    for (prevIt.GoToBegin(), currentIt.GoToBegin(); 
         !prevIt.IsAtEnd(); 
         ++prevIt, ++currentIt)
    {
      currentIt.Set((prevIt.Get()+currentIt.Get())/2.); 
    }
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[i]); 
    
    prevVelocityField = Superclass::DuplicateDeformableParameters(this->m_VelocityField[i]); 
  }
  if (timePoint == m_NumberOfVelocityField)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[m_NumberOfVelocityField-1]); 
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[m_NumberOfVelocityField]); 
  }
  
#endif
  
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::ReparameteriseTime()
{
  niftkitkDebugMacro(<< "ReparameteriseTime(): start");
  typedef ImageRegionIterator<DeformableParameterType> DeformableParametersIteratorType; 
  
  typename std::vector<double> length; 
  typename std::vector<double> velocity; 
  double totalLength = 0.; 
  double totalTime = 0.; 
  
  // Calculate the mean length and total mean length. 
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    DeformableParametersIteratorType iterator(this->m_VelocityField[i], this->m_VelocityField[i]->GetLargestPossibleRegion()); 
    double meanLength = 0.; 
    double meanVelocity = 0.; 
    int numberOfVoxels = 0; 
  
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      meanLength = (meanLength*numberOfVoxels)/(numberOfVoxels+1) + (iterator.Get().GetNorm()*this->m_TimeStep[i])/(numberOfVoxels+1); 
      meanVelocity = (meanVelocity*numberOfVoxels)/(numberOfVoxels+1) + (iterator.Get().GetNorm())/(numberOfVoxels+1); 
      numberOfVoxels++;    
    }
    
    niftkitkDebugMacro(<< "ReparameteriseTime(): meanLength=" << meanLength);
    niftkitkDebugMacro(<< "ReparameteriseTime(): meanVelocity=" << meanVelocity);
    length.push_back(meanLength); 
    velocity.push_back(meanVelocity); 
    totalLength += meanLength; 
    totalTime += this->m_TimeStep[i]; 
  }
    
  // Reparameterise the velocity field to be constant length. 
  double averageVelocity = totalLength/totalTime; 
  niftkitkDebugMacro(<< "ReparameteriseTime(): averageVelocity=" << averageVelocity);
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    DeformableParametersIteratorType iterator(this->m_VelocityField[i], this->m_VelocityField[i]->GetLargestPossibleRegion()); 
    
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      typename DeformableParameterType::PixelType value = iterator.Get(); 
      
      value = value*averageVelocity/velocity[i]; 
      iterator.Set(value); 
      
    }
    this->m_TimeStep[i] = this->m_TimeStep[i]*velocity[i]/averageVelocity; 
    niftkitkDebugMacro(<< "ReparameteriseTime(): this->m_TimeStep[i]=" << this->m_TimeStep[i]);
  }
  
  niftkitkDebugMacro(<< "ReparameteriseTime(): end");
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Shoot()
{
  typedef ImageRegionIterator<DeformableParameterType> DeformableParametersIteratorType; 
  typedef DisplacementFieldJacobianFilter<TDeformationScalar, TDeformationScalar, NDimensions> DisplacementFieldJacobianFilterType; 
  typedef MultiplyImageFilter<typename DisplacementFieldJacobianFilterType::OutputImageType, typename DisplacementFieldJacobianFilterType::OutputImageType, typename DisplacementFieldJacobianFilterType::OutputImageType> MultiplyJacobianImageFilterType; 
  typedef MultiplyImageFilter<typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType, typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType, typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType> MultiplyJacobianDeterminantImageFilterType; 
  typedef MultiplyImageFilter<typename Superclass::JacobianDeterminantFilterType::OutputImageType, typename Superclass::JacobianDeterminantFilterType::OutputImageType, typename Superclass::JacobianDeterminantFilterType::OutputImageType> MultiplySuperclassJacobianDeterminantImageFilterType; 
  
  // Filter for calculating Jacobian matrix from the displacement field. 
  typename DisplacementFieldJacobianFilterType::Pointer displacementFieldJacobianFilter = DisplacementFieldJacobianFilterType::New(); 
  // Filter for multiplying two Jacobian matrices. 
  typename MultiplyJacobianImageFilterType::Pointer multiplyJacobianImageFilter = MultiplyJacobianImageFilterType::New(); 
  // Filter for multiplying two Jacobian determinant.  
  typename MultiplyJacobianDeterminantImageFilterType::Pointer multiplyJacobianDeterminantImageFilter = MultiplyJacobianDeterminantImageFilterType::New(); 
  // Filter for calculating Jacobian of the fixed image displacement field. 
  typename Superclass::JacobianDeterminantFilterType::Pointer fixedImageJacobianDeterminantFilter = Superclass::JacobianDeterminantFilterType::New(); 
  // Filter for multiplying two Jacobian determinant of the superclass (double type). 
  typename MultiplySuperclassJacobianDeterminantImageFilterType::Pointer multiplySuperclassJacobianDeterminantImageFilter = MultiplySuperclassJacobianDeterminantImageFilterType::New(); 
  // The initial momentum which will be calculated from the initial velocity. 
  typename MomentumImageType::Pointer initialMomentum; 
  // The composed moving image Jacobian matrix. 
  typename DisplacementFieldJacobianFilterType::OutputImageType::Pointer movingImageJacobian; 
  // The composed moving image Jacobian determinant. 
  typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType::Pointer movingImageJacobianDeterminant; 
  // The composed fixed image Jacobian determinant. 
  typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType::Pointer fixedImageJacobianDeterminant; 
  
  this->m_ForwardJacobianImage = NULL; 
  this->m_BackwardJacobianImage = NULL; 
  
  // Convert the velocity to momentunm. 
  this->m_FluidPDESolver->SetInput(this->m_VelocityField[0]);
  this->m_FluidPDESolver->SetIsComputeVelcoity(false); 
  this->m_FluidPDESolver->Update(); 
  initialMomentum = this->m_FluidPDESolver->GetOutput(); 
  initialMomentum->DisconnectPipeline(); 
  double maxNorm = 0.; 
  
  // Copy image dimensions.
  typename Superclass::DeformationFieldSpacingType spacing = initialMomentum->GetSpacing();
  typename Superclass::DeformationFieldDirectionType direction = initialMomentum->GetDirection();
  typename Superclass::DeformationFieldOriginType origin = initialMomentum->GetOrigin();
  typename Superclass::DeformationFieldSizeType size = initialMomentum->GetLargestPossibleRegion().GetSize();
  typename Superclass::DeformationFieldIndexType index = initialMomentum->GetLargestPossibleRegion().GetIndex();
  typename Superclass::DeformationFieldRegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  typename MomentumImageType::SizeType regionSize = initialMomentum->GetLargestPossibleRegion().GetSize(); 
  typename MomentumImageType::Pointer currentMomemtum = Superclass::DuplicateDeformableParameters(initialMomentum); 
  
  // Set the transforms to 0. 
  typename DeformationFieldType::PixelType fieldValue;
  fieldValue.Fill(0);
  this->m_DeformationField->FillBuffer(fieldValue);
  this->m_FixedImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
  this->m_MovingImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
  
  typename DeformationFieldType::Pointer currentVelcoityField = Superclass::DuplicateDeformableParameters(this->m_VelocityField[0]); 
  typename DeformationFieldType::Pointer temp; 
  
  for (int i = 0; i < this->m_NumberOfVelocityField; i++)
  {
    // Generate the velocity from the momentum. 
    this->m_FluidPDESolver->SetInput(currentMomemtum);
    this->m_FluidPDESolver->SetIsComputeVelcoity(true); 
    this->m_FluidPDESolver->UpdateLargestPossibleRegion(); 
    currentVelcoityField = this->m_FluidPDESolver->GetOutput(); 
    currentVelcoityField->DisconnectPipeline(); 
    
    // Compose the moving image transform by "pulling back". Serious tidying later. 
    temp = Superclass::DuplicateDeformableParameters(currentVelcoityField);  
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[i]); 
    
    // Compose the Jacobian matrix of the moving image transform. 
    displacementFieldJacobianFilter->SetInput(this->m_MovingImageDeformationField); 
    displacementFieldJacobianFilter->Update(); 
    movingImageJacobian = displacementFieldJacobianFilter->GetOutput(); 
    movingImageJacobian->DisconnectPipeline(); 
    displacementFieldJacobianFilter->SetInput(this->m_MovingImageDeformationField); 
    displacementFieldJacobianFilter->Update(); 
    movingImageJacobianDeterminant = displacementFieldJacobianFilter->GetDeterminant(); 
    movingImageJacobianDeterminant->DisconnectPipeline(); 
    
    // Compose the fixed image transform by "pushing forward". Serious tidying later. 
    temp = Superclass::DuplicateDeformableParameters(this->m_FixedImageDeformationField); 
    this->m_FixedImageDeformationField = Superclass::DuplicateDeformableParameters(currentVelcoityField); 
    DeformableParametersIteratorType iterator(this->m_FixedImageDeformationField, this->m_FixedImageDeformationField->GetLargestPossibleRegion()); 
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      iterator.Set(iterator.Get()*-1.); 
    }
    fixedImageJacobianDeterminantFilter->SetInput(this->m_FixedImageDeformationField); 
    fixedImageJacobianDeterminantFilter->Update(); 
    typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer currentFixedImageJacobianDeterminant = fixedImageJacobianDeterminantFilter->GetOutput(); 
    currentFixedImageJacobianDeterminant->DisconnectPipeline(); 
    UpdateRegriddedDeformationParameters(this->m_FixedImageDeformationField, temp, +1.*this->m_TimeStep[i]); 
    
    // Compose Jacobian of the forward deformation. 
    if (this->m_ForwardJacobianImage.IsNull())
    {
      this->m_ForwardJacobianImage = currentFixedImageJacobianDeterminant; 
    }
    else
    {
      multiplySuperclassJacobianDeterminantImageFilter->SetInput1(currentFixedImageJacobianDeterminant); 
      multiplySuperclassJacobianDeterminantImageFilter->SetInput2(this->m_ForwardJacobianImage); 
      multiplySuperclassJacobianDeterminantImageFilter->Update(); 
      this->m_ForwardJacobianImage = multiplySuperclassJacobianDeterminantImageFilter->GetOutput(); 
      this->m_ForwardJacobianImage->DisconnectPipeline(); 
    }
        
    // Transform the initial momentum by the moving image transform. 
    typedef VectorResampleImageFilter<MomentumImageType, MomentumImageType> VectorResampleImageFilterType; 
    typename VectorResampleImageFilterType::Pointer momentumResampleFilter = VectorResampleImageFilterType::New(); 
    typedef VectorLinearInterpolateImageFunction<MomentumImageType, double>  InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    
    momentumResampleFilter->SetInput(initialMomentum); 
    this->UseMovingImageDeformationField(); 
    momentumResampleFilter->SetTransform(this); 
    momentumResampleFilter->SetOutputSpacing(spacing);
    momentumResampleFilter->SetOutputOrigin(origin); 
    momentumResampleFilter->SetOutputDirection(direction); 
    momentumResampleFilter->SetInterpolator(interpolator);      
    momentumResampleFilter->SetSize(regionSize); 
    momentumResampleFilter->Update(); 
    currentMomemtum = momentumResampleFilter->GetOutput(); 
    currentMomemtum->DisconnectPipeline(); 
    
    // Compute the Jacobian determinant. 
    //this->UseMovingImageDeformationField(); 
    //this->ComputeMinJacobian(); 
    //typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer movingTransformJacobian = this->GetJacobianImage(); 
    //movingTransformJacobian->DisconnectPipeline(); 
    
    // Multiply by the transpose of the Jacobian matrix and the Jacobian determinant. 
    typedef ImageRegionIterator<typename DisplacementFieldJacobianFilterType::OutputImageType> MovingImageJacobianIteratorType;
    typedef ImageRegionIterator<typename DisplacementFieldJacobianFilterType::OutputDeterminantImageType> JacobianImageIteratorType; 
    MovingImageJacobianIteratorType movingImageJacobianIterator(movingImageJacobian, movingImageJacobian->GetRequestedRegion()); 
    DeformableParametersIteratorType currentMomemtumIterator(currentMomemtum, currentMomemtum->GetLargestPossibleRegion()); 
    JacobianImageIteratorType movingImageJacobianDeterminantIterator(movingImageJacobianDeterminant, movingImageJacobianDeterminant->GetRequestedRegion()); 
    maxNorm = 0.; 
        
    for (movingImageJacobianIterator.GoToBegin(), currentMomemtumIterator.GoToBegin(), movingImageJacobianDeterminantIterator.GoToBegin(); 
         !movingImageJacobianIterator.IsAtEnd();       
         ++movingImageJacobianIterator, ++currentMomemtumIterator, ++movingImageJacobianDeterminantIterator)
    {
      typename MomentumImageType::PixelType oldValue = currentMomemtumIterator.Get(); 
      typename MomentumImageType::PixelType newValue; 
      typename DisplacementFieldJacobianFilterType::OutputPixelType jacobian = movingImageJacobianIterator.Get(); 
            
      newValue.Fill(0.); 
      // Transpose of the Jacobian matrix. 
      for (int m = 0; m < TFixedImage::ImageDimension; m++)
      {
        for (int n = 0; n < TFixedImage::ImageDimension; n++)
        {
          newValue[m] += jacobian(n,m)*oldValue[n]; 
        }
      }
      // Jacobian determinant. 
      newValue *= movingImageJacobianDeterminantIterator.Get(); 
      if (newValue.GetNorm() > maxNorm)
        maxNorm = newValue.GetNorm(); 
      currentMomemtumIterator.Set(newValue); 
    }
  }  
  
  
  niftkitkDebugMacro(<< "Shoot(): done");
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::AccumulateBackwardDeformationFromVelocityField(int timePoint)
{
  int index = timePoint-1; 
  
  if (index >= 0)
  {
    typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[index]); 
    UpdateRegriddedDeformationParameters(this->m_MovingImageDeformationField, temp, +1.*this->m_TimeStep[index]); 
  }
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::AccumulateForwardDeformationFromVelocityField(int timePoint)
{
  typedef ImageRegionIterator<DeformableParameterType> DeformableParametersIteratorType; 
  typename DeformationFieldType::Pointer temp = Superclass::DuplicateDeformableParameters(this->m_VelocityField[timePoint]); 
  this->m_DeformationField = temp; 
  DeformableParametersIteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion()); 
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(iterator.Get()*-1.); 
  }
  UpdateRegriddedDeformationParameters(this->m_FixedImageDeformationField, temp, this->m_TimeStep[timePoint]); 
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::ResetDeformationFields()
{
  typename DeformationFieldType::PixelType fieldValue;
  fieldValue.Fill(0);
  this->m_DeformationField->FillBuffer(fieldValue);
  this->m_FixedImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
  this->m_MovingImageDeformationField = Superclass::DuplicateDeformableParameters(this->m_DeformationField); 
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SaveField(typename DeformationFieldType::Pointer field, std::string filename)
{
  typedef ImageFileWriter<DeformableParameterType> ImageFileWriterType; 
  typename ImageFileWriterType::Pointer writer = ImageFileWriterType::New(); 
  
  writer->SetInput(field); 
  writer->SetFileName(filename); 
  writer->Update(); 
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
VelocityFieldDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::LoadField(typename DeformationFieldType::Pointer* field, std::string filename)
{
  typedef ImageFileReader<DeformableParameterType> ImageFileReaderType; 
  typename ImageFileReaderType::Pointer reader = ImageFileReaderType::New(); 
  
  try 
  {
    reader->SetFileName(filename); 
    reader->Update(); 
    (*field) = reader->GetOutput(); 
    (*field)->DisconnectPipeline(); 
  }
  catch (itk::ExceptionObject &e)
  {
    //field = DuplicateDeformableParameters(this->m_DeformationField); 
    //DeformationFieldPixelType fieldValue;
    //fieldValue.Fill(0);
    //field->FillBuffer(fieldValue);
  }
}






} // namespace itk.

#endif /*ITKVelocityFieldDeformableTransform_TXX_*/



























