/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-12 11:01:24 +0100 (Wed, 12 Oct 2011) $
 Revision          : $Revision: 7496 $
 Last modified by  : $Author: kkl $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKFLUIDDEFORMABLETRANSFORM_TXX_
#define ITKFLUIDDEFORMABLETRANSFORM_TXX_

#include "itkFluidDeformableTransform.h" 
#include "itkVectorResampleImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkScaleTransform.h"
#include "itkIdentityTransform.h"
#include "itkImageDuplicator.h"
#include "itkVectorImageToImageAdaptor.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkDisplacementFieldJacobianVectorFilter.h"
#include <limits>
#include "itkLogHelper.h"

namespace itk
{
// Constructor with default arguments
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::FluidDeformableTransform()
{
  niftkitkDebugMacro(<< "FluidDeformableTransform():Constructed");
  return;
}

// Destructor
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::~FluidDeformableTransform()
{
  niftkitkDebugMacro(<< "FluidDeformableTransform():Destroyed");
  return;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  // Superclass one will do.
  Superclass::PrintSelf(os,indent);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{

  // Setup deformation field.
  Superclass::Initialize(image);

  // Parameters array should match grid, not the fixed image.
  // this->ResizeParametersArray(this->m_DeformationField);
  
  this->SetIdentity();   
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetIdentity( void )
{
  // This resets parameters and deformation field. Thats all we need.
  Superclass::SetIdentity();  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InterpolateNextGrid(FixedImagePointer image)
{
  // niftkitkDebugMacro(<< "InterpolateNextGrid():Starting");
  
  typedef VectorResampleImageFilter< DeformationFieldType, DeformationFieldType > VectorResampleImageFilterType; 
  typename VectorResampleImageFilterType::Pointer vectorResampleImageFilter  = VectorResampleImageFilterType::New();
  typedef VectorLinearInterpolateImageFunction< DeformationFieldType, double >  InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  typedef IdentityTransform< double, NDimensions > IdentityTransform;
  typename IdentityTransform::Pointer identityTransform = IdentityTransform::New();
  
  ImageRegionIterator< DeformationFieldType > deformationIterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());
  typename DeformationFieldType::SpacingType currentDeformationSpacing = this->m_DeformationField->GetSpacing();
  typename TFixedImage::SpacingType newImageSpacing = image->GetSpacing(); 
      
  // Adjust the deformation to the new spacing/resolution because the deformation is in voxel unit. 
  for (deformationIterator.GoToBegin(); !deformationIterator.IsAtEnd(); ++deformationIterator)
  {
    DeformationFieldPixelType deformation = deformationIterator.Get(); 
    
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      deformation[i] = deformation[i]*currentDeformationSpacing[i]/newImageSpacing[i]; 
    }
    deformationIterator.Set(deformation);
  }
  
  //niftkitkDebugMacro(<< "InterpolateNextGrid():m_DeformationField");
  //niftkitkDebugMacro(<< this->m_DeformationField->GetSpacing());
  //niftkitkDebugMacro(<< this->m_DeformationField->GetOrigin());
  //niftkitkDebugMacro(<< "InterpolateNextGrid():image");
  //niftkitkDebugMacro(<< image->GetSpacing());
  //niftkitkDebugMacro(<< image->GetOrigin());
  
  vectorResampleImageFilter->SetInterpolator(interpolator);
  vectorResampleImageFilter->SetTransform(identityTransform);
  vectorResampleImageFilter->SetOutputDirection(image->GetDirection());
  vectorResampleImageFilter->SetOutputOrigin(image->GetOrigin());
  vectorResampleImageFilter->SetOutputSpacing(image->GetSpacing());
  vectorResampleImageFilter->SetSize(image->GetLargestPossibleRegion().GetSize());
  vectorResampleImageFilter->SetInput(this->m_DeformationField);
  //niftkitkDebugMacro(<< "InterpolateNextGrid():before Update");
  vectorResampleImageFilter->Update(); 
      
  this->m_DeformationField = vectorResampleImageFilter->GetOutput(); 
  this->m_DeformationField->DisconnectPipeline(); 
  
  // Make sure that the boundary condition still valid. 
  ImageRegionIteratorWithIndex< DeformationFieldType > iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  typename DeformationFieldType::SizeType size = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  DeformationFieldPixelType zeroDeformation; 
  
  for (unsigned int i = 0; i < NDimensions; i++)
    zeroDeformation[i] = 0.0; 
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    typename DeformationFieldType::IndexType index = iterator.GetIndex(); 
    
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      if (index[i] == 0 || index[i] == static_cast<typename DeformationFieldType::IndexType::IndexValueType>(size[i])-1)
      {
        iterator.Set(zeroDeformation); 
        break; 
      }
    }
  }
  
  // niftkitkDebugMacro(<< "InterpolateNextGrid():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::UpdateRegriddedDeformationParameters(DeformableParameterPointerType regriddedParameters, DeformableParameterPointerType currentPosition, double factor)
{
  // Compose the new regridded parameters from the existing regridded parameters and the current parameters. 
  // Need to perform linear interpolation because the displacement vectors are not pointing to the centres of the voxels. 
  // niftkitkDebugMacro(<< "FluidDeformableTransform::UpdateRegriddedDeformationParameters - Started");
  
  SetDeformableParameters(regriddedParameters); 
  this->ExtractComponents(); 
    
  // 1. Set up m_DeformationField from the current position.   
  SetDeformableParameters(currentPosition); 
  
  // 2. Compose the new regridded paramters using the linearly interpolated the regridded deformation. 
  // typedef BSplineInterpolateImageFunction< typename Superclass::DeformationFieldComponentImageType, double > InterpolatorType; 
  typedef LinearInterpolateImageFunction< typename Superclass::DeformationFieldComponentImageType, double > InterpolatorType; 
  typename InterpolatorType::Pointer interpolator[NDimensions]; 
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    interpolator[i] = InterpolatorType::New();
    interpolator[i]->SetInputImage(this->m_DeformationFieldComponent[i]); 
  }
  
  ImageRegionIteratorWithIndex< DeformationFieldType > iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  typedef Point< TDeformationScalar, NDimensions > DeformationOutputPointType; 
  typedef Point< double, NDimensions > DoubleDeformationOutputPointType; 
  typename DeformationFieldType::RegionType region = this->m_DeformationField->GetLargestPossibleRegion(); 
  typename DeformationFieldType::SizeType size = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  typename DeformationFieldType::PointType origin = this->m_DeformationField->GetOrigin();
  
  // niftkitkDebugMacro(<< "FluidDeformableTransform::UpdateRegriddedDeformationParameters - Looping");
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    typename DeformationFieldType::IndexType index = iterator.GetIndex(); 
    
#if 1  
    // Skip the boundary. 
    bool isBoundary = false; 
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      if (index[i] == 0 || index[i] == static_cast<typename DeformationFieldType::IndexType::IndexValueType>(size[i])-1)
      {
        isBoundary = true; 
        break; 
      }
    }
    if (isBoundary)
      continue; 
#endif    
    
    // Follow the current position. 
    DeformationOutputPointType physicalPoint; 
    DeformationOutputPointType physicalDeformation; 
    DoubleDeformationOutputPointType deformedPhysicalPoint; 
    this->m_DeformationField->TransformIndexToPhysicalPoint(index, physicalPoint); 
    ContinuousIndex< TDeformationScalar, NDimensions > imageDeformation;
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      imageDeformation[i] = iterator.Get()[i];
    }
    this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(imageDeformation, physicalDeformation);
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      deformedPhysicalPoint[i] = physicalPoint[i] + (physicalDeformation[i]-origin[i]); 
    }
    // Interpolate the regridded deformation and compose it with the current position.  
    DeformationFieldPixelType composedDeformation; 
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      if (interpolator[j]->IsInsideBuffer(deformedPhysicalPoint))
      {
        composedDeformation[j] = factor*imageDeformation[j] + interpolator[j]->Evaluate(deformedPhysicalPoint); 
      }
    }
    iterator.Set(composedDeformation);
  }
  
  ImageRegionIteratorWithIndex< DeformationFieldType > thisIterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  ImageRegionIteratorWithIndex< DeformationFieldType > regriddedIterator(regriddedParameters, regriddedParameters->GetLargestPossibleRegion());  
  
  for (thisIterator.GoToBegin(), regriddedIterator.GoToBegin(); 
       !thisIterator.IsAtEnd(); 
       ++thisIterator, ++regriddedIterator)
  {
    regriddedIterator.Set(thisIterator.Get()); 
  }
  
  // Try to release the memory.... if we can.
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_DeformationFieldComponent[i] = NULL; 
  }
  
  // niftkitkDebugMacro(<< "FluidDeformableTransform::UpdateRegriddedDeformationParameters - Finished");
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::DeformableParameterPointerType
FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::DuplicateDeformableParameters(const DeformableParameterType* deformableParameters)
{
  typedef ImageDuplicator<DeformableParameterType> ImageDuplicatorType; 
  typename ImageDuplicatorType::Pointer duplicator = ImageDuplicatorType::New(); 
  duplicator->SetInputImage(deformableParameters); 
  duplicator->Update(); 
  DeformableParameterPointerType copy = duplicator->GetOutput(); 
  copy->DisconnectPipeline(); 

  return copy; 
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetDeformableParameters(DeformableParameterPointerType parameters)
{
  this->m_DeformationField = parameters;
  this->Modified();
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
bool
FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::IsIdentity()
{
  ImageRegionConstIterator<DeformationFieldType> iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  
  niftkitkDebugMacro(<< "FluidDeformableTransform::IsIdentity(): epsilon=" << std::numeric_limits<TDeformationScalar>::epsilon());
  
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      if (fabs(iterator.Get()[i]) > std::numeric_limits<TDeformationScalar>::epsilon())
        return false; 
    }
  }
  return true; 
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InvertUsingGradientDescent(typename Self::Pointer invertedTransform, unsigned int maxIteration, double tol)
{
  // Create the necesssary filters. 
  const unsigned int NJDimensions = NDimensions*NDimensions; 
  typedef DisplacementFieldJacobianVectorFilter<TDeformationScalar, TDeformationScalar, NDimensions, NJDimensions> DisplacementFieldJacobianVectorFilterType; 
  typename DisplacementFieldJacobianVectorFilterType::Pointer displacementFieldJacobianVectorFilter = DisplacementFieldJacobianVectorFilterType::New(); 
  typedef VectorResampleImageFilter<typename DisplacementFieldJacobianVectorFilterType::OutputImageType, 
                                    typename DisplacementFieldJacobianVectorFilterType::OutputImageType, 
                                    double> VectorResampleImageFilterType; 
  typename VectorResampleImageFilterType::Pointer vectorResampleImageFilter = VectorResampleImageFilterType::New(); 
  typedef VectorLinearInterpolateImageFunction< typename DisplacementFieldJacobianVectorFilterType::OutputImageType, double >  InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New(); 
  
  // Save a copy of the original deformation field. 
  DeformableParameterPointerType originalParameters = DuplicateDeformableParameters(this->m_DeformationField); 
  
  // Compute the Jacobian. 
  displacementFieldJacobianVectorFilter->SetInput(this->m_DeformationField); 
  displacementFieldJacobianVectorFilter->Update(); 
  
  // Initialise inverse transform. 
  ImageRegionIterator< DeformationFieldType > inverseIterator(invertedTransform->GetDeformableParameters(), invertedTransform->GetDeformableParameters()->GetLargestPossibleRegion());  
  ImageRegionIterator< DeformationFieldType > originalIterator(originalParameters, originalParameters->GetLargestPossibleRegion());  

  for (inverseIterator.GoToBegin(), originalIterator.GoToBegin(); 
       !inverseIterator.IsAtEnd(); 
       ++inverseIterator, ++originalIterator)
  {
    inverseIterator.Set(-originalIterator.Get()); 
  }
  
  bool stop = false; 
  unsigned int iteration = 0; 
  while (!stop && iteration < maxIteration)
  {
    DeformableParameterPointerType duplicateInvertParameters = DuplicateDeformableParameters(invertedTransform->GetDeformableParameters()); 
    DeformableParameterPointerType originalCopyParameters = DuplicateDeformableParameters(originalParameters); 
    UpdateRegriddedDeformationParameters(originalCopyParameters, duplicateInvertParameters, 1.); 
    
    ImageRegionIterator< DeformationFieldType > currentIterator(originalCopyParameters, originalCopyParameters->GetLargestPossibleRegion());  
    
    // Resample the Jacobian according to the current inverse transform.                                 
    vectorResampleImageFilter->SetInput(displacementFieldJacobianVectorFilter->GetOutput()); 
    vectorResampleImageFilter->SetInterpolator(interpolator);
    vectorResampleImageFilter->SetTransform(invertedTransform); 
    vectorResampleImageFilter->SetOutputDirection(displacementFieldJacobianVectorFilter->GetOutput()->GetDirection());
    vectorResampleImageFilter->SetOutputOrigin(displacementFieldJacobianVectorFilter->GetOutput()->GetOrigin());
    vectorResampleImageFilter->SetOutputSpacing(displacementFieldJacobianVectorFilter->GetOutput()->GetSpacing());
    vectorResampleImageFilter->SetSize(displacementFieldJacobianVectorFilter->GetOutput()->GetLargestPossibleRegion().GetSize());
    vectorResampleImageFilter->Update(); 
    
    ImageRegionIterator< typename VectorResampleImageFilterType::OutputImageType > jacobianIterator(vectorResampleImageFilter->GetOutput(), vectorResampleImageFilter->GetOutput()->GetLargestPossibleRegion());  
    
    stop = true; 
    for (currentIterator.GoToBegin(), jacobianIterator.GoToBegin(), inverseIterator.GoToBegin(); 
         !currentIterator.IsAtEnd(); 
         ++currentIterator, ++jacobianIterator, ++inverseIterator)
    {
      double norm = currentIterator.Get().GetNorm(); 
      if (norm > tol)
        stop = false; 
      
      Matrix<float, NDimensions, NDimensions> matrix; 
      
      for (unsigned int i = 0; i < NDimensions; i++)
      {
        for (unsigned int j = 0; j < NDimensions; j++)
        {
          unsigned int index = j + i*NDimensions; 
          //matrix(i, j) = jacobianIterator.Get()[index]; 
          // Get the transpose. 
          matrix(j, i) = jacobianIterator.Get()[index]; 
        }
      }
      
      Vector<float, NDimensions> gradient = matrix*currentIterator.Get(); 
      
      inverseIterator.Set(inverseIterator.Get() - gradient*0.08); 
    }
    invertedTransform->Modified(); 
    
    iteration++; 
  }    
  this->SetDeformableParameters(originalParameters); 
  
  std::cout << "iteration=" << iteration << std::endl; 
    
}




template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeSquareRoot(typename Self::Pointer sqrtTransform, unsigned int maxIteration, double tol)
{
  // Create the necesssary filters. 
  const unsigned int NJDimensions = NDimensions*NDimensions; 
  typedef DisplacementFieldJacobianVectorFilter<TDeformationScalar, TDeformationScalar, NDimensions, NJDimensions> DisplacementFieldJacobianVectorFilterType; 
  typename DisplacementFieldJacobianVectorFilterType::Pointer displacementFieldJacobianVectorFilter = DisplacementFieldJacobianVectorFilterType::New(); 
  typedef VectorResampleImageFilter<typename DisplacementFieldJacobianVectorFilterType::OutputImageType, 
                                    typename DisplacementFieldJacobianVectorFilterType::OutputImageType, 
                                    double> VectorResampleImageFilterType; 
  typename VectorResampleImageFilterType::Pointer vectorResampleImageFilter = VectorResampleImageFilterType::New(); 
  typedef VectorLinearInterpolateImageFunction< typename DisplacementFieldJacobianVectorFilterType::OutputImageType, double >  InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New(); 
  
  // Save a copy of the original deformation field. 
  DeformableParameterPointerType originalParameters = DuplicateDeformableParameters(this->m_DeformationField); 
  
  // Initialise sqrt transform. 
  ImageRegionIterator< DeformationFieldType > originalIterator(originalParameters, originalParameters->GetLargestPossibleRegion());  
  ImageRegionIterator<DeformationFieldType> sqrtIterator(sqrtTransform->GetDeformationField(), sqrtTransform->GetDeformationField()->GetLargestPossibleRegion()); 

  for (sqrtIterator.GoToBegin(), originalIterator.GoToBegin(); 
       !sqrtIterator.IsAtEnd(); 
       ++sqrtIterator, ++originalIterator)
  {
    sqrtIterator.Set(originalIterator.Get()/2.); 
  }
  
  
  bool stop = false; 
  unsigned int iteration = 0; 
  while (!stop && iteration < maxIteration)
  {
    // Compute the Jacobian. 
    displacementFieldJacobianVectorFilter->SetInput(sqrtTransform->GetDeformationField()); 
    displacementFieldJacobianVectorFilter->Update(); 
  
    // Resample the Jacobian according to the current sqrt transform.                                 
    vectorResampleImageFilter->SetInput(displacementFieldJacobianVectorFilter->GetOutput()); 
    vectorResampleImageFilter->SetInterpolator(interpolator);
    vectorResampleImageFilter->SetTransform(sqrtTransform); 
    vectorResampleImageFilter->SetOutputDirection(displacementFieldJacobianVectorFilter->GetOutput()->GetDirection());
    vectorResampleImageFilter->SetOutputOrigin(displacementFieldJacobianVectorFilter->GetOutput()->GetOrigin());
    vectorResampleImageFilter->SetOutputSpacing(displacementFieldJacobianVectorFilter->GetOutput()->GetSpacing());
    vectorResampleImageFilter->SetSize(displacementFieldJacobianVectorFilter->GetOutput()->GetLargestPossibleRegion().GetSize());
    vectorResampleImageFilter->Update(); 
    
    // Compose the sqrt transform with itself. 
    DeformableParameterPointerType duplicateSqrtParameters = DuplicateDeformableParameters(sqrtTransform->GetDeformableParameters()); 
    DeformableParameterPointerType composedSqrtParameters = DuplicateDeformableParameters(sqrtTransform->GetDeformableParameters()); 
    UpdateRegriddedDeformationParameters(composedSqrtParameters, duplicateSqrtParameters, 1.); 
    
    
    // Get the inverse. 
    typename Self::Pointer inverseTransform = Self::New();
    inverseTransform->SetDeformableParameters(duplicateSqrtParameters); 
    sqrtTransform->InvertUsingGradientDescent(inverseTransform.GetPointer(), 200, tol); 
    
    // Calculate the determinant of the Jacobian of the inverse transform. 
    inverseTransform->ComputeMaxJacobian(); 
    
    // Compose original deformation field with inverse sqrt deformation field. 
    DeformableParameterPointerType duplicateOriginalParameters = DuplicateDeformableParameters(originalParameters); 
    DeformableParameterPointerType duplicateInverseParameters = DuplicateDeformableParameters(inverseTransform->GetDeformableParameters()); 
    UpdateRegriddedDeformationParameters(duplicateOriginalParameters, duplicateInverseParameters, 1.); 
    
    // Calculate the gradient. 
    ImageRegionIterator<typename VectorResampleImageFilterType::OutputImageType > jacobianIterator(vectorResampleImageFilter->GetOutput(), vectorResampleImageFilter->GetOutput()->GetLargestPossibleRegion());  
    ImageRegionIterator<DeformationFieldType> composedTTIterator(composedSqrtParameters, composedSqrtParameters->GetLargestPossibleRegion()); 
    ImageRegionIterator<DeformationFieldType> originalIterator(originalParameters, originalParameters->GetLargestPossibleRegion()); 
    ImageRegionIterator<DeformationFieldType> composedOTIterator(duplicateOriginalParameters, duplicateOriginalParameters->GetLargestPossibleRegion()); 
    ImageRegionIterator<typename Superclass::JacobianDeterminantFilterType::OutputImageType> determinantIterator(this->m_JacobianFilter->GetOutput(), this->m_JacobianFilter->GetOutput()->GetLargestPossibleRegion()); 
            
    stop = true; 
    double maxDiff = 0.; 
    double maxGradient = 0.; 
    for (sqrtIterator.GoToBegin(), jacobianIterator.GoToBegin(), originalIterator.GoToBegin(), composedTTIterator.GoToBegin(), composedOTIterator.GoToBegin(), determinantIterator.GoToBegin(); 
         !sqrtIterator.IsAtEnd(); 
         ++sqrtIterator, ++jacobianIterator, ++originalIterator, ++composedTTIterator, ++composedOTIterator, ++determinantIterator)
    {
      Matrix<float, NDimensions, NDimensions> matrix; 
      
      for (unsigned int i = 0; i < NDimensions; i++)
      {
        for (unsigned int j = 0; j < NDimensions; j++)
        {
          unsigned int index = j + i*NDimensions; 
          //matrix(i, j) = jacobianIterator.Get()[index]; 
          // Get the transpose. 
          matrix(j, i) = jacobianIterator.Get()[index]; 
        }
      }
      
      Vector<float, NDimensions> diff = composedTTIterator.Get()-originalIterator.Get(); 
      Vector<float, NDimensions> gradient = matrix*diff + determinantIterator.Get()*(sqrtIterator.Get() - composedOTIterator.Get()); 
      
      if (diff.GetNorm() > tol)                                            
      {
        stop = false;
      }
      sqrtIterator.Set(sqrtIterator.Get() - gradient*0.1); 
      
      if (diff.GetNorm() > maxDiff)
        maxDiff = diff.GetNorm(); 
      if (gradient.GetNorm() > maxGradient)
        maxGradient = gradient.GetNorm(); 
      
    }
    sqrtTransform->Modified(); 
    iteration++;
     
    std::cout << "maxDiff=" << maxDiff << ", maxGradient=" << maxGradient << std::endl; 
  }
  
  
  this->SetDeformableParameters(originalParameters); 
  
}




} // namespace itk.

#endif /*ITKFLUIDDEFORMABLETRANSFORM_TXX_*/




