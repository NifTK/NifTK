/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFluidVelocityToDeformationFilter_txx
#define __itkFluidVelocityToDeformationFilter_txx

#include "itkFluidVelocityToDeformationFilter.h"
#include <itkConstNeighborhoodIterator.h>
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkDerivativeOperator.h>
#include <itkVectorNeighborhoodInnerProduct.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::FluidVelocityToDeformationFilter()
{
  m_MaxDeformation = 0.0;
  m_IsNegativeVelocity = false; 
  m_InputMask = NULL; 
  m_IsTakeDerivative = true; 
}

template <class TScalarType, unsigned int NDimensions>
void
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::SetNthInput(unsigned int idx, const InputImageType *image)
{
  this->ProcessObject::SetNthInput(idx, const_cast< InputImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetNthInput():Set input[" << idx << "] to address:" << image);
}

template <class TScalarType, unsigned int NDimensions>
void
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <class TScalarType, unsigned int NDimensions>
void
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 2 inputs.
  if (numberOfInputs != 2)
    {
      itkExceptionMacro(<< "FluidVelocityToDeformationFilter should only have 2 inputs.");
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
}

template <class TScalarType, unsigned int NDimensions>
void
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, ThreadIdType threadNumber)
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  typename InputImageType::ConstPointer currentDeformationField 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::ConstPointer fluidVelocityField 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  // Pointer to output image.
  typename OutputImageType::Pointer outputDeformationField 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
    
  // Prepare the operator for doing the partial derivatives. 
  typedef DerivativeOperator < TScalarType, NDimensions > DerivativeOperatorType;  
  DerivativeOperatorType derivativeOperator[NDimensions];
  
  for (unsigned int index = 0; index < NDimensions; index++)
  {
    derivativeOperator[index].SetOrder(1);
    derivativeOperator[index].SetDirection(index);
    derivativeOperator[index].CreateDirectional();
    derivativeOperator[index].FlipAxes();
  }

  // Iterator for current deformation field.
  typedef ConstNeighborhoodIterator< InputImageType > NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  
  NeighborhoodIteratorType deformationIterator(radius, currentDeformationField, outputRegionForThread);

  // Iterator for fluid velocity.
  typedef ImageRegionConstIterator< InputImageType > VelocityFieldIteratorType;
  VelocityFieldIteratorType velocityIterator(fluidVelocityField, outputRegionForThread);

  // Iterator for output
  typedef ImageRegionIterator< OutputImageType > DeformationFieldIteratorType;
  DeformationFieldIteratorType newDeformationFieldIterator(outputDeformationField, outputRegionForThread);  

  typedef VectorNeighborhoodInnerProduct < InputImageType > InnerProductType; 
  InnerProductType innerProduct; 
  const typename OutputImageType::SizeType regionSize = outputDeformationField->GetLargestPossibleRegion().GetSize();
  
  // Update the deformation field using the material derivative. 
  deformationIterator.SetNeedToUseBoundaryCondition(false);

  // Create a zero deformation for boundary.
  typename OutputImageType::PixelType zeroDeformation;
  for (unsigned int index = 0; index < NDimensions; index++)
    {
      zeroDeformation[index] = 0;
    }

  deformationIterator.GoToBegin();
  velocityIterator.GoToBegin();
  newDeformationFieldIterator.GoToBegin();

  
  for ( ; !deformationIterator.IsAtEnd(); ++deformationIterator, ++velocityIterator, ++newDeformationFieldIterator)
  {
    typename InputImageType::PixelType deformation = deformationIterator.GetCenterPixel();
    const typename InputImageType::IndexType position = deformationIterator.GetIndex();
        
    typename InputImageType::PixelType velocity = velocityIterator.Get();

    bool isBoundary = false;
    typename OutputImageType::PixelType deltaDeformation; 
    
    // Bounds checking.
    for (unsigned int index = 0; index < NDimensions; index++)
    {
      if (position[index] <= 0 || position[index] >= static_cast<typename OutputImageType::IndexType::IndexValueType>(regionSize[index]-1))
      {
        isBoundary = true;
        break;
      }
    }
    if (isBoundary)
      {
        newDeformationFieldIterator.Set(zeroDeformation);
      }
    else
      {
        if (this->m_IsNegativeVelocity)
        {
          for (unsigned int index = 0; index < NDimensions; index++)
            velocity[index] = -velocity[index]; 
        }
      
        deltaDeformation = velocity;
    
        if (m_IsTakeDerivative)
        {
          for (unsigned int index = 0; index < NDimensions; index++)
            {
              deltaDeformation -= velocity[index]*innerProduct(deformationIterator.GetSlice(index), deformationIterator, derivativeOperator[index]);
            }
        }
    
        newDeformationFieldIterator.Set(deltaDeformation);
      }    
  }
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


template <class TScalarType, unsigned int NDimensions>
void
FluidVelocityToDeformationFilter<TScalarType, NDimensions>
::AfterThreadedGenerateData()
{
  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Starting:");
  this->m_MaxDeformation = 0.0;
  
  typedef ImageRegionConstIterator< OutputImageType > DeformationFieldIteratorType;
  DeformationFieldIteratorType deformationFieldIterator(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());  
  InputImageType* input = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  typedef ImageRegionConstIterator<InputImageMaskType> MaskIteratorType; 
  MaskIteratorType* maskaskIterator = NULL;
  if (this->m_InputMask != NULL)
  {
    maskaskIterator = new MaskIteratorType(this->m_InputMask, this->m_InputMask->GetLargestPossibleRegion());
    maskaskIterator->GoToBegin(); 
  }
  
  for (deformationFieldIterator.GoToBegin(); !deformationFieldIterator.IsAtEnd(); ++deformationFieldIterator)
  {
    typename OutputImageType::PixelType deformation = deformationFieldIterator.Get();
    double deformationNorm = 0.0;
    
    typename OutputImageType::PointType physicalPoint; 
    input->TransformIndexToPhysicalPoint(deformationFieldIterator.GetIndex(), physicalPoint); 
    if (maskaskIterator != NULL)
    {
      if (maskaskIterator->Get() < 128)
      {
        ++(*maskaskIterator); 
        continue; 
      }
      ++(*maskaskIterator); 
    }
    
    for (unsigned int index = 0; index < NDimensions; index++)
    {
      deformationNorm += deformation[index]*deformation[index]; 
    }
    deformationNorm = sqrt(deformationNorm);
    if (deformationNorm > this->m_MaxDeformation)
    {
      this->m_MaxDeformation = deformationNorm; 
    }
  }
  
  delete maskaskIterator; 
  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Finished: m_MaxDeformation=" << m_MaxDeformation);
}



} // end namespace itk

#endif
