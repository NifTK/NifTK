/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
  //typedef BSplineInterpolateImageFunction< typename Superclass::DeformationFieldComponentImageType, double > InterpolatorType; 
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
    
#if 0
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
      else
      {
        composedDeformation[j] = 0.; 
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
  DeformableParameterPointerType tempParameters = this->m_DeformationField; 
  this->m_DeformationField = originalParameters; 
  originalParameters = tempParameters; 
  
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
  invertedTransform->Modified(); 
  
  bool stop = false; 
  double maxDiff = 0.;
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
    maxDiff = 0.;
    double totalDiff = 0.; 
    int totalNumber = 0;     
    for (currentIterator.GoToBegin(), jacobianIterator.GoToBegin(), inverseIterator.GoToBegin(), originalIterator.GoToBegin(); 
         !currentIterator.IsAtEnd(); 
         ++currentIterator, ++jacobianIterator, ++inverseIterator, ++originalIterator)
    {
      double norm = currentIterator.Get().GetNorm(); 
      if (norm > tol)
        stop = false; 
      if (norm > maxDiff)
        maxDiff = norm; 
      
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
      
      inverseIterator.Set(inverseIterator.Get() - gradient*0.5); 
      
      if (originalIterator.Get().GetNorm() > 1.e-7)
      {
        totalDiff += norm/originalIterator.Get().GetNorm(); 
        totalNumber++; 
      }
    }
    
    invertedTransform->Modified(); 
    iteration++; 
    std::cout << "iteration=" << iteration << ", maxDiff=" << maxDiff << ", averageDiff=" << totalDiff/totalNumber << std::endl; 
    
    // std::cout << "maxInverseDiff=" << maxInverseDiff << std::endl; 
  }    
  this->SetDeformableParameters(originalParameters); 
  
  //std::cout << "iteration=" << iteration << ", maxDiff=" << maxDiff << std::endl; 
    
}




template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FluidDeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeSquareRoot(typename Self::Pointer sqrtTransform, unsigned int maxInverseIteration, unsigned int maxIteration, double tol)
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
  DeformableParameterPointerType originalParameters = this->m_DeformationField; 
  this->m_DeformationField = DuplicateDeformableParameters(this->m_DeformationField); 
  
  // Initialise sqrt transform. 
  ImageRegionIterator<DeformationFieldType> originalIterator(originalParameters, originalParameters->GetLargestPossibleRegion());  
  ImageRegionIteratorWithIndex<DeformationFieldType> sqrtIterator(sqrtTransform->GetDeformationField(), sqrtTransform->GetDeformationField()->GetLargestPossibleRegion()); 

  for (sqrtIterator.GoToBegin(), originalIterator.GoToBegin(); 
       !sqrtIterator.IsAtEnd(); 
       ++sqrtIterator, ++originalIterator)
  {
    sqrtIterator.Set(originalIterator.Get()/2.); 
  }
  sqrtTransform->Modified(); 
  
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
    DeformableParameterPointerType inverseParameters = DuplicateDeformableParameters(sqrtTransform->GetDeformableParameters()); 
    inverseTransform->SetDeformableParameters(inverseParameters); 
    sqrtTransform->InvertUsingGradientDescent(inverseTransform.GetPointer(), maxInverseIteration, tol); 
    
#if 0    
    // Test.     
    DeformableParameterPointerType test1 = DuplicateDeformableParameters(inverseTransform->GetDeformableParameters()); 
    DeformableParameterPointerType test2 = DuplicateDeformableParameters(composedSqrtParameters); 
    DeformableParameterPointerType test3 = DuplicateDeformableParameters(sqrtTransform->GetDeformableParameters()); 
    
    typename DeformationFieldType::IndexType index; 
    index[0] = 254;  
    index[1] = 254; 
    
    std::cout << "test1=" << test1->GetPixel(index) << ", test2=" << test2->GetPixel(index) << ",test3=" << test3->GetPixel(index) << std::endl; 
    
    UpdateRegriddedDeformationParameters(test3, test1, 1.); 
    std::cout << "test1=" << test1->GetPixel(index) << ", test2=" << test2->GetPixel(index) << ",test3=" << test3->GetPixel(index) << std::endl; 
    std::cout << "inverseParameters=" << inverseParameters->GetPixel(index) << ", sqrtTransform=" << sqrtTransform->GetDeformableParameters()->GetPixel(index) << std::endl; 
#endif                     
    
    // Calculate the determinant of the Jacobian of the inverse transform. 
    inverseTransform->ComputeMaxJacobian(); 
    
    // Compose original deformation field with inverse sqrt deformation field. 
    DeformableParameterPointerType duplicateOriginalParameters = DuplicateDeformableParameters(originalParameters); 
    DeformableParameterPointerType duplicateInverseParameters = DuplicateDeformableParameters(inverseTransform->GetDeformableParameters()); 
    
    //std::cout << "1 duplicateOriginalParameters=" << duplicateOriginalParameters->GetPixel(index) << ", duplicateInverseParameters=" << duplicateInverseParameters->GetPixel(index) << ", sqrtTransform=" << sqrtTransform->GetDeformableParameters()->GetPixel(index) << std::endl; 
    UpdateRegriddedDeformationParameters(duplicateOriginalParameters, duplicateInverseParameters, 1.); 
    //std::cout << "2 duplicateOriginalParameters=" << duplicateOriginalParameters->GetPixel(index) << ", duplicateInverseParameters=" << duplicateInverseParameters->GetPixel(index) << std::endl; 
    
    // Calculate the gradient. 
    ImageRegionIterator<typename VectorResampleImageFilterType::OutputImageType > jacobianIterator(vectorResampleImageFilter->GetOutput(), vectorResampleImageFilter->GetOutput()->GetLargestPossibleRegion());  
    ImageRegionIterator<DeformationFieldType> composedTTIterator(composedSqrtParameters, composedSqrtParameters->GetLargestPossibleRegion()); 
    ImageRegionIterator<DeformationFieldType> composedOTIterator(duplicateOriginalParameters, duplicateOriginalParameters->GetLargestPossibleRegion()); 
    ImageRegionIterator<typename Superclass::JacobianDeterminantFilterType::OutputImageType> determinantIterator(inverseTransform->m_JacobianFilter->GetOutput(), inverseTransform->m_JacobianFilter->GetOutput()->GetLargestPossibleRegion()); 
            
    stop = true; 
    double factor = 1.; 
    double maxDiff = 0.; 
    double maxGradient = 0.; 
    double maxInverseDiff = 0.; 
    
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
      Vector<float, NDimensions> inverseDiff = sqrtIterator.Get()-composedOTIterator.Get(); 
      Vector<float, NDimensions> gradient = matrix*diff; //  + inverseDiff*determinantIterator.Get(); 
      
      if (gradient.GetNorm() > maxGradient)
        maxGradient = gradient.GetNorm(); 
    }
    
    factor = 1.; 
#if 0    
    if (maxGradient > 0.25)
      factor = 0.25/maxGradient; 
    factor = std::min<double>(factor, 0.25); 
#else
    if (maxGradient > 0.5)
      factor = 0.5/maxGradient; 
    factor = std::min<double>(factor, 0.5); 
#endif    
    std::cout << "factor=" << factor << std::endl; 
    
    double totalDiff = 0.; 
    double totalInverseDiff = 0.; 
    int totalNumber = 0; 
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
      Vector<float, NDimensions> inverseDiff = sqrtIterator.Get()-composedOTIterator.Get(); 
      Vector<float, NDimensions> gradient = matrix*diff; //  + inverseDiff*determinantIterator.Get(); 
      
      if (diff.GetNorm() > tol)                                            
      {
        stop = false;
      }
      sqrtIterator.Set(sqrtIterator.Get() - gradient*factor); 
      
      if (diff.GetNorm() > maxDiff)
      {
        maxDiff = diff.GetNorm(); 
        //std::cout << "index=" << sqrtIterator.GetIndex() << ",sqrt=" << sqrtIterator.Get() << ", maxDiff=" << maxDiff << "," << composedTTIterator.Get() << "," << originalIterator.Get() <<  std::endl; 
      }
      if (gradient.GetNorm() > maxGradient)
        maxGradient = gradient.GetNorm(); 
      if (inverseDiff.GetNorm() > maxInverseDiff)
        maxInverseDiff = inverseDiff.GetNorm(); 
      
      if (originalIterator.Get().GetNorm() > 1.e-7)
      {
        totalDiff += diff.GetNorm()/originalIterator.Get().GetNorm(); 
        totalInverseDiff += inverseDiff.GetNorm()/originalIterator.Get().GetNorm(); 
        totalNumber++; 
      }
    }
    sqrtTransform->Modified(); 
    iteration++;
    
    std::cout << "maxDiff=" << maxDiff << ", maxGradient=" << maxGradient << ", maxInverseDiff=" << maxInverseDiff << std::endl; 
    std::cout << "averageDiff=" << totalDiff/totalNumber << ", averageInverseDiff=" << totalInverseDiff/totalNumber  << std::endl; 
  }
  
  
  this->SetDeformableParameters(originalParameters); 
  
}




} // namespace itk.

#endif /*ITKFLUIDDEFORMABLETRANSFORM_TXX_*/




